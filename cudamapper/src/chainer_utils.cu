/*
* Copyright 2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "chainer_utils.cuh"

#include <fstream>
#include <sstream>
#include <cstdlib>

// Needed for accumulate - remove when ported to cuda
#include <numeric>
#include <limits>

#include <cub/cub.cuh>
#include <thrust/execution_policy.h>

#include <claraparabricks/genomeworks/utils/cudautils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{
namespace chainerutils
{

#define BLOCK_COUNT 1792
#define BLOCK_SIZE 64
#define WARP_SIZE 32

__device__ bool operator==(const QueryTargetPair& a, const QueryTargetPair& b)
{
    return a.query_read_id_ == b.query_read_id_ && a.target_read_id_ == b.target_read_id_;
}

__device__ bool operator==(const QueryReadID& a, const QueryReadID& b)
{
    return a.query_read_id_ == b.query_read_id_;
}

struct QueryAnchorPos
{
    read_id_t query_read_id_;
    position_in_read_t query_position_in_read_;
};

struct AnchorToQueryAnchorPos
{
    __device__ __forceinline__ QueryAnchorPos operator()(const Anchor& a) const
    {
        QueryAnchorPos q;
        q.query_read_id_          = a.query_read_id_;
        q.query_position_in_read_ = a.query_position_in_read_;
        return q;
    }
};

__device__ bool operator==(const QueryAnchorPos& a, const QueryAnchorPos& b)
{
    return a.query_read_id_ == b.query_read_id_ && a.query_position_in_read_ == b.query_position_in_read_;
}

__device__ Overlap create_simple_overlap(const Anchor& start, const Anchor& end, const int32_t num_anchors)
{
    Overlap overlap;
    overlap.num_residues_ = num_anchors;

    overlap.query_read_id_  = start.query_read_id_;
    overlap.target_read_id_ = start.target_read_id_;
    assert(start.query_read_id_ == end.query_read_id_ && start.target_read_id_ == end.target_read_id_);

    overlap.query_start_position_in_read_ = min(start.query_position_in_read_, end.query_position_in_read_);
    overlap.query_end_position_in_read_   = max(start.query_position_in_read_, end.query_position_in_read_);
    bool is_negative_strand               = end.target_position_in_read_ < start.target_position_in_read_;
    if (is_negative_strand)
    {
        overlap.relative_strand                = RelativeStrand::Reverse;
        overlap.target_start_position_in_read_ = end.target_position_in_read_;
        overlap.target_end_position_in_read_   = start.target_position_in_read_;
    }
    else
    {
        overlap.relative_strand                = RelativeStrand::Forward;
        overlap.target_start_position_in_read_ = start.target_position_in_read_;
        overlap.target_end_position_in_read_   = end.target_position_in_read_;
    }
    return overlap;
}

Overlap create_simple_overlap_cpu(const Anchor& start, const Anchor& end, const int32_t num_anchors)
{
    Overlap overlap;
    overlap.num_residues_ = num_anchors;

    overlap.query_read_id_  = start.query_read_id_;
    overlap.target_read_id_ = start.target_read_id_;
    assert(start.query_read_id_ == end.query_read_id_ && start.target_read_id_ == end.target_read_id_);

    overlap.query_start_position_in_read_ = min(start.query_position_in_read_, end.query_position_in_read_);
    overlap.query_end_position_in_read_   = max(start.query_position_in_read_, end.query_position_in_read_);
    bool is_negative_strand               = end.target_position_in_read_ < start.target_position_in_read_;
    if (is_negative_strand)
    {
        overlap.relative_strand                = RelativeStrand::Reverse;
        overlap.target_start_position_in_read_ = end.target_position_in_read_;
        overlap.target_end_position_in_read_   = start.target_position_in_read_;
    }
    else
    {
        overlap.relative_strand                = RelativeStrand::Forward;
        overlap.target_start_position_in_read_ = start.target_position_in_read_;
        overlap.target_end_position_in_read_   = end.target_position_in_read_;
    }
    return overlap;
}

void drop_overlaps_by_mask(device_buffer<Overlap>& d_overlaps,
                           device_buffer<bool>& d_mask,
                           const std::int32_t n_overlaps,
                           device_buffer<Overlap>& d_dest,
                           device_buffer<int32_t>& d_filtered_count,
                           DefaultDeviceAllocator& allocator,
                           cudaStream_t& cuda_stream)
{
    device_buffer<char> d_temp_buf(allocator, cuda_stream);
    void* d_temp_storage           = nullptr;
    std::size_t temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_overlaps.data(),
                               d_mask.data(),
                               d_dest.data(),
                               d_filtered_count.data(),
                               n_overlaps,
                               cuda_stream);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_overlaps.data(),
                               d_mask.data(),
                               d_dest.data(),
                               d_filtered_count.data(),
                               n_overlaps,
                               cuda_stream);
    int32_t n_filtered = cudautils::get_value_from_device(d_filtered_count.data(), cuda_stream);
    d_mask.clear_and_resize(n_filtered);
    chainerutils::initialize_mask<<<BLOCK_COUNT, BLOCK_SIZE, 0, cuda_stream>>>(d_mask.data(), true, n_filtered);
}

__global__ void mask_repeat_runs(const int32_t* const run_starts,
                                 const int32_t* const run_lengths,
                                 bool* mask,
                                 const int32_t max_chain_length,
                                 const int64_t num_chains)
{
    const std::size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride        = blockDim.x * gridDim.x;
    for (int i = thread_id; i < num_chains; i += stride)
    {
        int32_t chain_start  = run_starts[i];
        int32_t chain_length = run_lengths[i];
        const bool mask_val  = chain_length <= max_chain_length;
        for (int32_t anchor_index = chain_start; anchor_index < (chain_start + chain_length); ++anchor_index)
        {
            mask[anchor_index] = mask_val;
        }
    }
}

device_buffer<Anchor> mask_repeated_anchors(const device_buffer<Anchor>& anchors,
                                            const int32_t max_run_length,
                                            int64_t& num_filtered_anchors,
                                            DefaultDeviceAllocator& allocator,
                                            cudaStream_t& cuda_stream)
{

    device_buffer<QueryAnchorPos> qpos_runs(anchors.size(), allocator, cuda_stream);
    device_buffer<int32_t> qpos_run_lengths(anchors.size(), allocator, cuda_stream);

    device_buffer<int32_t> num_qpos_runs_d(1, allocator, cuda_stream);

    void* temp_storage_d           = nullptr;
    std::size_t temp_storage_bytes = 0;

    AnchorToQueryAnchorPos anchor_to_pos_op;
    cub::TransformInputIterator<QueryAnchorPos, AnchorToQueryAnchorPos, const Anchor*> query_positions_d(anchors.data(), anchor_to_pos_op);

    // Run length encode the query positions of each anchor.
    // Repeats appear as runs of identical query positions
    cub::DeviceRunLengthEncode::Encode(temp_storage_d,
                                       temp_storage_bytes,
                                       query_positions_d,
                                       qpos_runs.data(),
                                       qpos_run_lengths.data(),
                                       num_qpos_runs_d.data(),
                                       anchors.size(),
                                       cuda_stream);
    device_buffer<char> temp_buf_d(temp_storage_bytes, allocator, cuda_stream);
    temp_storage_d = temp_buf_d.data();
    cub::DeviceRunLengthEncode::Encode(temp_storage_d,
                                       temp_storage_bytes,
                                       query_positions_d,
                                       qpos_runs.data(),
                                       qpos_run_lengths.data(),
                                       num_qpos_runs_d.data(),
                                       anchors.size(),
                                       cuda_stream);

    // Total number of query-pos struct runs.
    int32_t num_query_pos_runs = cudautils::get_value_from_device(num_qpos_runs_d.data(), cuda_stream);

    auto thrust_exec_policy = thrust::cuda::par(allocator).on(cuda_stream);
    device_buffer<int32_t> qpos_run_starts(num_query_pos_runs, allocator, cuda_stream);

    // Run an exclusive prefix sum to calculate the starts of each
    // query-position run in the anchors array.
    thrust::exclusive_scan(thrust_exec_policy,
                           qpos_run_lengths.begin(),
                           qpos_run_lengths.end(),
                           qpos_run_starts.begin());

    device_buffer<bool> mask(anchors.size(), allocator, cuda_stream);
    initialize_mask<<<BLOCK_COUNT, BLOCK_SIZE, 0, cuda_stream>>>(mask.data(), true, anchors.size());

    // could be replaced by a thrust::copy_if,
    // but not sure how to use a predicate with an argument from a second array.
    // We also need to trim both the start and length arrays.
    mask_repeat_runs<<<BLOCK_COUNT, BLOCK_SIZE, 0, cuda_stream>>>(qpos_run_starts.data(),
                                                                  qpos_run_lengths.data(),
                                                                  mask.data(),
                                                                  max_run_length,
                                                                  num_query_pos_runs);

    device_buffer<Anchor> filtered_anchors(anchors.size(), allocator, cuda_stream);
    device_buffer<int64_t> num_filtered_anchors_d(1, allocator, cuda_stream);

    // // Drop anchors that have been masked because they are considered part of a repeat run.
    temp_storage_d     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(temp_storage_d,
                               temp_storage_bytes,
                               anchors.data(),
                               mask.data(),
                               filtered_anchors.data(),
                               num_filtered_anchors_d.data(),
                               anchors.size(),
                               cuda_stream);
    temp_buf_d.clear_and_resize(temp_storage_bytes);
    temp_storage_d = temp_buf_d.data();
    cub::DeviceSelect::Flagged(temp_storage_d,
                               temp_storage_bytes,
                               anchors.data(),
                               mask.data(),
                               filtered_anchors.data(),
                               num_filtered_anchors_d.data(),
                               anchors.size(),
                               cuda_stream);

    num_filtered_anchors = cudautils::get_value_from_device(num_filtered_anchors_d.data(), cuda_stream);

    return filtered_anchors;
}

// TODO VI: this may have some thread overwrite issues, as well as some problems
// uninitialized variables in overlap struct.
// This also has an upper bound on how many anchors we actually process. If num_anchors is greater
// than 1792 * 32, we never actually process that anchor
__global__ void backtrace_anchors_to_overlaps(const Anchor* anchors,
                                              Overlap* overlaps,
                                              double* scores,
                                              bool* max_select_mask,
                                              int32_t* predecessors,
                                              const int32_t n_anchors,
                                              const int32_t min_score)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride  = blockDim.x * gridDim.x;
    for (int i = tid; i < n_anchors; i += stride)
    {
        double score = scores[i];
        if (score >= min_score)
        {
            int32_t index                = i;
            int32_t first_index          = index;
            int32_t num_anchors_in_chain = 0;
            Anchor final_anchor          = anchors[i];

            while (index != -1)
            {
                first_index  = index;
                int32_t pred = predecessors[index];
                if (pred != -1)
                {
                    max_select_mask[pred] = false;
                }
                num_anchors_in_chain++;
                index = predecessors[index];
            }
            Anchor first_anchor = anchors[first_index];
            overlaps[i]         = create_simple_overlap(first_anchor, final_anchor, num_anchors_in_chain);
        }
        else
        {
            max_select_mask[i] = false;
        }
    }
}

__global__ void backtrace_anchors_to_overlaps_debug(const Anchor* anchors,
                                                    Overlap* overlaps,
                                                    double* scores,
                                                    bool* max_select_mask,
                                                    int32_t* predecessors,
                                                    const int32_t n_anchors,
                                                    const int32_t min_score)
{
    int32_t end = n_anchors - 1;
    for (int32_t i = end; i >= 0; i--)
    {
        if (scores[i] >= min_score)
        {
            int32_t index                = i;
            int32_t first_index          = index;
            int32_t num_anchors_in_chain = 0;
            Anchor final_anchor          = anchors[i];

            while (index != -1)
            {
                first_index  = index;
                int32_t pred = predecessors[index];
                if (pred != -1)
                {
                    max_select_mask[pred] = false;
                }
                num_anchors_in_chain++;
                index = predecessors[index];
            }
            Anchor first_anchor = anchors[first_index];
            overlaps[i]         = create_simple_overlap(first_anchor, final_anchor, num_anchors_in_chain);
        }
        else
        {
            max_select_mask[i] = false;
        }
    }
}
void backtrace_anchors_to_overlaps_cpu(const Anchor* anchors,
                                       Overlap* overlaps,
                                       double* scores,
                                       bool* max_select_mask,
                                       int32_t* predecessors,
                                       const int32_t n_anchors,
                                       const int32_t min_score)
{
    int32_t end = n_anchors - 1;
    for (int32_t i = end; i >= 0; i--)
    {
        if (scores[i] >= min_score)
        {
            int32_t index                = i;
            int32_t first_index          = index;
            int32_t num_anchors_in_chain = 0;
            Anchor final_anchor          = anchors[i];

            while (index != -1)
            {
                first_index  = index;
                int32_t pred = predecessors[index];
                if (pred != -1)
                {
                    max_select_mask[pred] = false;
                }
                num_anchors_in_chain++;
                index = predecessors[index];
            }
            Anchor first_anchor = anchors[first_index];
            overlaps[i]         = create_simple_overlap_cpu(first_anchor, final_anchor, num_anchors_in_chain);
        }
        else
        {
            max_select_mask[i] = false;
        }
    }
}
__global__ void convert_offsets_to_ends(std::int32_t* starts, std::int32_t* lengths, std::int32_t* ends, std::int32_t n_starts)
{
    std::int32_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride     = blockDim.x * gridDim.x;
    for (int i = d_tid; i < n_starts; i += stride)
    {
        ends[d_tid] = starts[d_tid] + lengths[d_tid];
    }
}

// we take each query that is encoded as length (ie there are 4 queries that have the same id)
// so we take the (how many queries are in that section) and divide it by the size of the tile to get the
// number of tiles for that query
__global__ void calculate_tiles_per_read(const std::int32_t* lengths,
                                         const int32_t num_reads,
                                         const int32_t tile_size,
                                         std::int32_t* tiles_per_read)
{
    int32_t d_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_thread_id < num_reads)
    {
        // if the number of queries in that section are not evenly divisible, we need an extra block
        // to accomadate for the leftovers
        int32_t n_integer_blocks    = lengths[d_thread_id] / tile_size;
        int32_t remainder           = lengths[d_thread_id] % tile_size;
        tiles_per_read[d_thread_id] = remainder == 0 ? n_integer_blocks : n_integer_blocks + 1;
    }
}

__global__ void calculate_tile_starts(const std::int32_t* query_starts,
                                      const std::int32_t* tiles_per_query,
                                      std::int32_t* tile_starts,
                                      const int32_t tile_size,
                                      const int32_t num_queries,
                                      const std::int32_t* tiles_per_query_up_to_point)
{
    int32_t d_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_thread_id < num_queries)
    {
        // for each tile, we look up the query it corresponds to and offset it by the which tile in the query
        // we're at multiplied by the total size of the tile
        for (int i = 0; i < tiles_per_query[d_thread_id]; i++)
        {
            //finds the offset in the ragged array and offsets to find start of "next" sub array
            tile_starts[tiles_per_query_up_to_point[d_thread_id] + i] = query_starts[d_thread_id] + (i * tile_size);
        }
    }
}

void encode_query_locations_from_anchors(device_buffer<Anchor>& anchors,
                                         device_buffer<int32_t>& query_starts,
                                         device_buffer<int32_t>& query_lengths,
                                         device_buffer<int32_t>& query_ends,
                                         const int64_t num_anchors,
                                         int32_t& n_queries,
                                         DefaultDeviceAllocator& _allocator,
                                         cudaStream_t& _cuda_stream)
{
    AnchorToQueryReadIDOp anchor_to_read_op;
    // This takes anchors and outputs and converts the anchors to QueryReadID types (references)
    cub::TransformInputIterator<QueryReadID, AnchorToQueryReadIDOp, const Anchor*> d_queries(anchors.data(), anchor_to_read_op);
    // create buffer of size number of anchors
    device_buffer<QueryReadID> d_query_read_ids(anchors.size(), _allocator, _cuda_stream);
    // vector of number of read ids...? I don't know what this is for
    // This is used to store the length of the encoded sequence
    device_buffer<int32_t> d_num_query_read_ids(1, _allocator, _cuda_stream);

    // don't know what this is for yet
    // this is for internal use for run length encoding
    void* d_temp_storage           = nullptr;
    std::size_t temp_storage_bytes = 0;

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_queries,
                                       d_query_read_ids.data(),
                                       query_lengths.data(),
                                       d_num_query_read_ids.data(),
                                       num_anchors,
                                       _cuda_stream);
    device_buffer<char> d_temp_buf(temp_storage_bytes, _allocator, _cuda_stream);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_queries,
                                       d_query_read_ids.data(),
                                       query_lengths.data(), // this is the vector of encoded lengths
                                       d_num_query_read_ids.data(),
                                       num_anchors,
                                       _cuda_stream);
    // this is just the "length" of the encoded sequence
    n_queries = cudautils::get_value_from_device(d_num_query_read_ids.data(), _cuda_stream);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_lengths.data(), // this is the vector of encoded lengths
                                  query_starts.data(),  // at this point, this vector is empty
                                  n_queries,
                                  _cuda_stream);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    // this gives us the number of queries up to that point. eg How many query starts we have at each index
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_lengths.data(), // this is the vector of encoded lengths
                                  query_starts.data(),
                                  n_queries,
                                  _cuda_stream);

    // paper uses the ends and finds the beginnings with x - w + 1, are we converting to that here?
    // TODO VI: I'm not entirely sure what this is for? I think we want to change the read query
    // (defined by [query_start, query_start + query_length] to [query_end - query_length + 1, query_end])
    // The above () is NOT true
    convert_offsets_to_ends<<<BLOCK_COUNT, BLOCK_SIZE, 0, _cuda_stream>>>(query_starts.data(),  // this gives how many starts at each index
                                                                          query_lengths.data(), // this is the vector of encoded lengths
                                                                          query_ends.data(),
                                                                          n_queries);
}

void encode_anchor_query_target_pairs(const Anchor* anchors,
                                      int32_t n_anchors,
                                      int32_t tile_size,
                                      device_buffer<int32_t>& query_target_starts,
                                      device_buffer<int32_t>& query_target_lengths,
                                      device_buffer<int32_t>& query_target_ends,
                                      device_buffer<int32_t>& tiles_per_read,
                                      int32_t& n_query_target_pairs,
                                      int32_t& n_qt_tiles,
                                      DefaultDeviceAllocator& _allocator,
                                      cudaStream_t& _cuda_stream,
                                      int32_t block_size)
{
    AnchorToQueryTargetPairOp qt_pair_op;
    cub::TransformInputIterator<QueryTargetPair, AnchorToQueryTargetPairOp, const Anchor*> d_query_target_pairs(anchors, qt_pair_op);
    device_buffer<QueryTargetPair> d_qt_pairs(n_anchors, _allocator, _cuda_stream);
    device_buffer<int32_t> d_num_query_target_pairs(1, _allocator, _cuda_stream);

    device_buffer<char> d_temp_buf(_allocator, _cuda_stream);
    void* d_temp_storage           = nullptr;
    std::size_t temp_storage_bytes = 0;

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_query_target_pairs,
                                       d_qt_pairs.data(),
                                       query_target_lengths.data(),
                                       d_num_query_target_pairs.data(),
                                       n_anchors,
                                       _cuda_stream);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_query_target_pairs,
                                       d_qt_pairs.data(),
                                       query_target_lengths.data(), // this is the vector of encoded lengths
                                       d_num_query_target_pairs.data(),
                                       n_anchors,
                                       _cuda_stream);

    n_query_target_pairs = cudautils::get_value_from_device(d_num_query_target_pairs.data(), _cuda_stream);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_target_lengths.data(),
                                  query_target_starts.data(),
                                  n_query_target_pairs,
                                  _cuda_stream);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_target_lengths.data(),
                                  query_target_starts.data(),
                                  n_query_target_pairs, _cuda_stream);

    convert_offsets_to_ends<<<(n_query_target_pairs / block_size) + 1, block_size, 0, _cuda_stream>>>(query_target_starts.data(),
                                                                                                      query_target_lengths.data(),
                                                                                                      query_target_ends.data(),
                                                                                                      n_query_target_pairs);

    if (tile_size > 0)
    {
        calculate_tiles_per_read<<<(n_query_target_pairs / block_size) + 1, 32, 0, _cuda_stream>>>(query_target_starts.data(), n_query_target_pairs, tile_size, tiles_per_read.data());
        device_buffer<int32_t> d_n_qt_tiles(1, _allocator, _cuda_stream);

        d_temp_storage     = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage,
                               temp_storage_bytes,
                               tiles_per_read.data(),
                               d_n_qt_tiles.data(),
                               n_query_target_pairs,
                               _cuda_stream);
        d_temp_buf.clear_and_resize(temp_storage_bytes);
        d_temp_storage = d_temp_buf.data();
        cub::DeviceReduce::Sum(d_temp_storage,
                               temp_storage_bytes,
                               tiles_per_read.data(),
                               d_n_qt_tiles.data(),
                               n_query_target_pairs,
                               _cuda_stream);
        n_qt_tiles = cudautils::get_value_from_device(d_n_qt_tiles.data(), _cuda_stream);
    }
}

void encode_overlap_query_target_pairs(Overlap* overlaps,
                                       int32_t n_overlaps,
                                       device_buffer<int32_t>& query_target_starts,
                                       device_buffer<int32_t>& query_target_lengths,
                                       device_buffer<int32_t>& query_target_ends,
                                       int32_t& n_query_target_pairs,
                                       DefaultDeviceAllocator& _allocator,
                                       cudaStream_t& _cuda_stream,
                                       int32_t block_size)
{
    OverlapToQueryTargetPairOp qt_pair_op;
    cub::TransformInputIterator<QueryTargetPair, OverlapToQueryTargetPairOp, Overlap*> d_query_target_pairs(overlaps, qt_pair_op);
    device_buffer<QueryTargetPair> d_qt_pairs(n_overlaps, _allocator, _cuda_stream);
    device_buffer<int32_t> d_num_query_target_pairs(1, _allocator, _cuda_stream);

    device_buffer<char> d_temp_buf(_allocator, _cuda_stream);
    void* d_temp_storage           = nullptr;
    std::size_t temp_storage_bytes = 0;

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_query_target_pairs,
                                       d_qt_pairs.data(),
                                       query_target_lengths.data(),
                                       d_num_query_target_pairs.data(),
                                       n_overlaps,
                                       _cuda_stream);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_query_target_pairs,
                                       d_qt_pairs.data(),
                                       query_target_lengths.data(),
                                       d_num_query_target_pairs.data(),
                                       n_overlaps,
                                       _cuda_stream);

    n_query_target_pairs = cudautils::get_value_from_device(d_num_query_target_pairs.data(), _cuda_stream);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_target_lengths.data(),
                                  query_target_starts.data(),
                                  n_query_target_pairs,
                                  _cuda_stream);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_target_lengths.data(),
                                  query_target_starts.data(),
                                  n_query_target_pairs,
                                  _cuda_stream);

    convert_offsets_to_ends<<<(n_query_target_pairs / block_size) + 1, block_size, 0, _cuda_stream>>>(query_target_starts.data(), query_target_lengths.data(), query_target_ends.data(), n_query_target_pairs);
}

__global__ void initialize_mask(bool* mask, bool val, const int32_t n_values)
{
    const int32_t d_tid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t grid_stride = blockDim.x * gridDim.x;
    for (int32_t i = d_tid; i < n_values; i += grid_stride)
    {
        mask[i] = val;
    }
}

__global__ void initialize_array(int32_t* array, int32_t value, const int32_t num_values)
{
    const int32_t d_tid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t grid_stride = blockDim.x * gridDim.x;
    for (int32_t i = d_tid; i < num_values; i += grid_stride)
    {
        array[i] = value;
    }
}

__global__ void initialize_array(double* array, double value, const int32_t num_values)
{
    const int32_t d_tid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t grid_stride = blockDim.x * gridDim.x;
    for (int32_t i = d_tid; i < num_values; i += grid_stride)
    {
        array[i] = value;
    }
}

} // namespace chainerutils
} // namespace cudamapper
} // namespace genomeworks
} // namespace claraparabricks

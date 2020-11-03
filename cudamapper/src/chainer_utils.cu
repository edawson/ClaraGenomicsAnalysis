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

__device__ bool operator==(const QueryTargetPair& a, const QueryTargetPair& b)
{
    return a.query_read_id_ == b.query_read_id_ && a.target_read_id_ == b.target_read_id_;
}

__device__ bool operator==(const QueryReadID& a, const QueryReadID& b)
{
    return a.query_read_id_ == b.query_read_id_;
}

__host__ __device__ Overlap create_simple_overlap(const Anchor& start, const Anchor& end, const int32_t num_anchors)
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
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_anchors)
    {

        int32_t global_overlap_index = d_tid;
        if (scores[d_tid] >= min_score)
        {

            int32_t index                = global_overlap_index;
            int32_t first_index          = index;
            int32_t num_anchors_in_chain = 0;
            Anchor final_anchor          = anchors[global_overlap_index];

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
            Anchor first_anchor            = anchors[first_index];
            overlaps[global_overlap_index] = create_simple_overlap(first_anchor, final_anchor, num_anchors_in_chain);
            // Overlap final_overlap          = overlaps[global_overlap_index];
            // printf("%d %d %d %d %d %d %d %f\n",
            //        final_overlap.query_read_id_, final_overlap.query_start_position_in_read_, final_overlap.query_end_position_in_read_,
            //        final_overlap.target_read_id_, final_overlap.target_start_position_in_read_, final_overlap.target_end_position_in_read_,
            //        final_overlap.num_residues_,
            //        final_score);
        }
        else
        {
            max_select_mask[global_overlap_index] = false;
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
            Anchor first_anchor            = anchors[first_index];
            overlaps[i] = create_simple_overlap(first_anchor, final_anchor, num_anchors_in_chain);
        }
        else
        {
            max_select_mask[i] = false;
        }
    }
}

void backtrace_anchors_to_overlaps_cpu(std::vector<Anchor>& anchors,
                                       std::vector<Overlap>& overlaps,
                                       std::vector<double>& scores,
                                       std::vector<bool>&  max_select_mask,
                                       std::vector<int32_t>& predecessors,
                                       const int32_t n_anchors,
                                       const int32_t min_score)
{
    for (int32_t i = overlaps.size() - 1; i >= 0; i--)
    {
        // we have a type mismatch comparison here
        //if (max_select_mask[i] == false)
        //    continue;
        if (scores[i] >= min_score)
        {
            int32_t index                = i;
            int32_t first_index          = index;
            int32_t num_anchors_in_chain = 1;
            Anchor final_anchor          = anchors[i];

            while (index != -1)
            {
                first_index  = index;
                const int32_t pred = predecessors[index];
                // I have a predecessor, so mark the overlap that ends at pred for filter
                if (pred != -1)
                {
                    max_select_mask[pred] = false;
                    num_anchors_in_chain++;
                }
                index = predecessors[index];
            }
            Anchor first_anchor            = anchors[first_index];
            overlaps[i] = create_simple_overlap(first_anchor, final_anchor, num_anchors_in_chain);
        }
        else
        {
            // we fail to meet min score, so mark for filter
            max_select_mask[i] = false;
        }
    }
}
__global__ void convert_offsets_to_ends(std::int32_t* starts, std::int32_t* lengths, std::int32_t* ends, std::int32_t n_starts)
{
    std::int32_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_starts)
    {
        ends[d_tid] = starts[d_tid] + lengths[d_tid] - 1;
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

void encode_anchor_query_locations(const Anchor* anchors,
                                   int32_t n_anchors,
                                   int32_t tile_size,
                                   device_buffer<int32_t>& query_starts,
                                   device_buffer<int32_t>& query_lengths,
                                   device_buffer<int32_t>& query_ends,
                                   device_buffer<int32_t>& tiles_per_query,
                                   device_buffer<int32_t>& tile_starts,
                                   int32_t& n_queries,
                                   int32_t& n_query_tiles,
                                   DefaultDeviceAllocator& _allocator,
                                   cudaStream_t& _cuda_stream,
                                   int32_t block_size)
{
    AnchorToQueryReadIDOp anchor_to_read_op;
    // This takes anchors and outputs and converts the anchors to QueryReadID types (references)
    cub::TransformInputIterator<QueryReadID, AnchorToQueryReadIDOp, const Anchor*> d_queries(anchors, anchor_to_read_op);
    // create buffer of size number of anchors
    device_buffer<QueryReadID> d_query_read_ids(n_anchors, _allocator, _cuda_stream);
    // vector of number of read ids...? I don't know what this is for
    // This is used to store the length of the encoded sequence
    device_buffer<int32_t> d_num_query_read_ids(1, _allocator, _cuda_stream);

    // don't know what this is for yet
    // this is for internal use for run length encoding
    device_buffer<char> d_temp_buf(_allocator, _cuda_stream);
    void* d_temp_storage           = nullptr;
    std::size_t temp_storage_bytes = 0;

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_queries,
                                       d_query_read_ids.data(),
                                       query_lengths.data(),
                                       d_num_query_read_ids.data(),
                                       n_anchors);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_queries,
                                       d_query_read_ids.data(),
                                       query_lengths.data(), // this is the vector of encoded lengths
                                       d_num_query_read_ids.data(),
                                       n_anchors);
    // this is just the "length" of the encoded sequence
    n_queries          = cudautils::get_value_from_device(d_num_query_read_ids.data(), _cuda_stream);
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
    convert_offsets_to_ends<<<(n_queries / block_size) + 1, block_size, 0, _cuda_stream>>>(query_starts.data(),  // this gives how many starts at each index
                                                                                           query_lengths.data(), // this is the vector of encoded lengths
                                                                                           query_ends.data(),
                                                                                           n_queries);

    // what case is this? A: This always runs because TILE_SIZE is fixed at 1024
    if (tile_size > 0)
    {
        calculate_tiles_per_read<<<(n_queries / block_size) + 1, block_size, 0, _cuda_stream>>>(query_lengths.data(), n_queries, tile_size, tiles_per_query.data());
        // This stores the total number of tiles that we need, calculated with the below Sum()
        device_buffer<int32_t> d_n_query_tiles(1, _allocator, _cuda_stream);

        d_temp_storage     = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, tiles_per_query.data(), d_n_query_tiles.data(), n_queries);

        d_temp_buf.clear_and_resize(temp_storage_bytes);
        d_temp_storage = d_temp_buf.data();
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, tiles_per_query.data(), d_n_query_tiles.data(), n_queries);

        // this is the length of tile_starts
        n_query_tiles = cudautils::get_value_from_device(d_n_query_tiles.data(), _cuda_stream);

        // This is used to calculate the offsets for tile_starts
        device_buffer<int32_t> d_tiles_per_query_up_to_point(n_queries, _allocator, _cuda_stream);

        d_temp_storage     = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      tiles_per_query.data(), // this is the vector of encoded lengths
                                      d_tiles_per_query_up_to_point.data(),
                                      n_queries,
                                      _cuda_stream);

        d_temp_buf.clear_and_resize(temp_storage_bytes);
        d_temp_storage = d_temp_buf.data();

        cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      tiles_per_query.data(), // this is the vector of encoded lengths
                                      d_tiles_per_query_up_to_point.data(),
                                      n_queries,
                                      _cuda_stream);

        calculate_tile_starts<<<(n_queries / block_size) + 1, block_size, 0, _cuda_stream>>>(query_starts.data(), tiles_per_query.data(), tile_starts.data(), tile_size, n_queries, d_tiles_per_query_up_to_point.data());
    }
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
                                       n_anchors);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_query_target_pairs,
                                       d_qt_pairs.data(),
                                       query_target_lengths.data(), // this is the vector of encoded lengths
                                       d_num_query_target_pairs.data(),
                                       n_anchors);

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
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, tiles_per_read.data(), d_n_qt_tiles.data(), n_query_target_pairs);
        d_temp_buf.clear_and_resize(temp_storage_bytes);
        d_temp_storage = d_temp_buf.data();
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, tiles_per_read.data(), d_n_qt_tiles.data(), n_query_target_pairs);
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
                                       n_overlaps);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_query_target_pairs,
                                       d_qt_pairs.data(),
                                       query_target_lengths.data(),
                                       d_num_query_target_pairs.data(),
                                       n_overlaps);

    n_query_target_pairs = cudautils::get_value_from_device(d_num_query_target_pairs.data(), _cuda_stream);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_target_lengths.data(),
                                  query_target_starts.data(),
                                  n_query_target_pairs, _cuda_stream);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_target_lengths.data(),
                                  query_target_starts.data(),
                                  n_query_target_pairs, _cuda_stream);

    convert_offsets_to_ends<<<(n_query_target_pairs / block_size) + 1, block_size, 0, _cuda_stream>>>(query_target_starts.data(), query_target_lengths.data(), query_target_ends.data(), n_query_target_pairs);
}

} // namespace chainerutils
} // namespace cudamapper
} // namespace genomeworks
} // namespace claraparabricks

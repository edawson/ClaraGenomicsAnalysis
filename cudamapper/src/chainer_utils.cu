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

__global__ void convert_offsets_to_ends(std::int32_t* starts, std::int32_t* lengths, std::int32_t* ends, std::int32_t n_starts)
{
    std::int32_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_starts)
    {
        ends[d_tid] = starts[d_tid] + lengths[d_tid] - 1;
    }
}

__global__ void calculate_tiles_per_read(std::int32_t* query_read_anchor_counts,
                                         const int32_t num_reads,
                                         const int32_t tile_size,
                                         std::int32_t* tiles_per_read)
{
    int32_t d_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_thread_id < num_reads)
    {
        int32_t n_integer_blocks    = query_read_anchor_counts[d_thread_id] / tile_size;
        int32_t remainder           = query_read_anchor_counts[d_thread_id] % tile_size;
        tiles_per_read[d_thread_id] = remainder == 0 ? n_integer_blocks : n_integer_blocks + 1;
    }
}

__global__ void calculate_tile_starts(std::int32_t* query_starts,
                                      std::int32_t* tiles_per_query,
                                      std::int32_t* tile_starts,
                                      const int32_t tile_size,
                                      int32_t num_queries)
{
    int32_t counter = 0;
    for (int32_t i = 0; i < num_queries; ++i)
    {
        for (int32_t j = 0; j < tiles_per_query[i]; ++j)
        {
            tile_starts[counter] = query_starts[i] + (j * tile_size);
        }
    }
}

int32_t count_unmasked(const bool* mask,
                       int32_t n_values,
                       DefaultDeviceAllocator& _allocator,
                       cudaStream_t& _cuda_stream)
{
    device_buffer<char> d_temp_buf(_allocator, _cuda_stream);
    void* d_temp_storage           = nullptr;
    std::size_t temp_storage_bytes = 0;

    device_buffer<int32_t> d_num_unmasked(1, _allocator, _cuda_stream);

    BoolToIntConverter converter_op;
    cub::TransformInputIterator<int32_t, BoolToIntConverter, const bool*> d_bool_ints(mask, converter_op);

    cub::DeviceReduce::Sum(d_temp_storage,
                           temp_storage_bytes,
                           d_bool_ints,
                           d_num_unmasked.data(),
                           n_values);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceReduce::Sum(d_temp_storage,
                           temp_storage_bytes,
                           d_bool_ints,
                           d_num_unmasked.data(),
                           n_values);
    int32_t num_unmasked = cudautils::get_value_from_device(d_num_unmasked.data(), _cuda_stream);
    return num_unmasked;
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

__device__ __forceinline__ Overlap empty_overlap()
{
    Overlap empty;
    empty.query_read_id_                 = 0;
    empty.query_start_position_in_read_  = 0;
    empty.query_end_position_in_read_    = 0;
    empty.target_read_id_                = 0;
    empty.target_start_position_in_read_ = 0;
    empty.target_end_position_in_read_   = 0;
    empty.num_residues_                  = 0;
    return empty;
}

__global__ void chain_anchors_by_backtrace(const Anchor* anchors,
                                           Overlap* overlaps,
                                           double* scores,
                                           bool* max_select_mask,
                                           int32_t* predecessors,
                                           const int32_t n_anchors,
                                           const int32_t min_score)
{
    const int32_t d_tid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t grid_stride = blockDim.x * gridDim.x;
    for (int32_t i = d_tid; i < n_anchors; i += grid_stride)
    {

        int32_t global_overlap_index = i;
        if (scores[i] >= min_score)
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

__global__ void set_mask_values(bool* mask, int32_t n_values, const bool value)
{
    const int32_t d_tid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t grid_stride = blockDim.x * gridDim.x;
    for (int32_t i = d_tid; i < n_values; i += grid_stride)
    {
        mask[i] = value;
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
    cub::TransformInputIterator<QueryReadID, AnchorToQueryReadIDOp, const Anchor*> d_queries(anchors, anchor_to_read_op);
    device_buffer<QueryReadID> d_query_read_ids(n_anchors, _allocator, _cuda_stream);
    device_buffer<int32_t> d_num_query_read_ids(1, _allocator, _cuda_stream);

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
                                       query_lengths.data(),
                                       d_num_query_read_ids.data(),
                                       n_anchors);

    n_queries          = cudautils::get_value_from_device(d_num_query_read_ids.data(), _cuda_stream);
    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_lengths.data(),
                                  query_starts.data(),
                                  n_queries,
                                  _cuda_stream);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_lengths.data(),
                                  query_starts.data(),
                                  n_queries,
                                  _cuda_stream);

    convert_offsets_to_ends<<<(n_queries / block_size) + 1, block_size, 0, _cuda_stream>>>(query_starts.data(),
                                                                                           query_lengths.data(),
                                                                                           query_ends.data(),
                                                                                           n_queries);

// #define DEBUG_QUERY_LOCATIONS
#ifdef DEBUG_QUERY_LOCATIONS
    std::vector<int32_t> q_starts;
    std::vector<int32_t> q_ends;
    q_starts.resize(n_queries);
    q_ends.resize(n_queries);
    cudautils::device_copy_n(query_starts.data(), n_queries, q_starts.data(), _cuda_stream);
    cudautils::device_copy_n(query_ends.data(), n_queries, q_ends.data(), _cuda_stream);
    for (size_t i = 0; i < q_starts.size(); ++i)
    {
        std::cout << i << " " << q_starts[i] << " " << q_ends[i] << std::endl;
    }
#endif

    if (tile_size > 0)
    {
        calculate_tiles_per_read<<<(n_queries / block_size) + 1, 32, 0, _cuda_stream>>>(query_lengths.data(), n_queries, tile_size, tiles_per_query.data());
        device_buffer<int32_t> d_n_query_tiles(1, _allocator, _cuda_stream);

        d_temp_storage     = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, tiles_per_query.data(), d_n_query_tiles.data(), n_queries);
        d_temp_buf.clear_and_resize(temp_storage_bytes);
        d_temp_storage = d_temp_buf.data();
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, tiles_per_query.data(), d_n_query_tiles.data(), n_queries);

        n_query_tiles = cudautils::get_value_from_device(d_n_query_tiles.data(), _cuda_stream);
        calculate_tile_starts<<<1, 1, 0, _cuda_stream>>>(query_starts.data(), tiles_per_query.data(), tile_starts.data(), tile_size, n_queries);
        std::cerr << "Generated " << n_query_tiles << " tiles." << std::endl;
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
                                       query_target_lengths.data(),
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
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

#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>

#include <thrust/execution_policy.h>
#include <thrust/transform_scan.h>
#include <thrust/reduce.h>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{
namespace chainerutils
{

__device__ __forceinline__ Anchor empty_anchor()
{
    Anchor a;
    a.query_read_id_           = UINT32_MAX;
    a.query_position_in_read_  = UINT32_MAX;
    a.target_read_id_          = UINT32_MAX;
    a.target_position_in_read_ = UINT32_MAX;
    return a;
}
struct ConvertOverlapToNumResidues : public thrust::unary_function<Overlap, int32_t>
{
    __host__ __device__ int32_t operator()(const Overlap& o) const
    {
        return o.num_residues_;
    }
};

__host__ __device__ Overlap create_overlap(const Anchor& start, const Anchor& end, const int32_t num_anchors)
{
    Overlap overlap;
    overlap.num_residues_ = num_anchors;

    overlap.query_read_id_  = start.query_read_id_;
    overlap.target_read_id_ = start.target_read_id_;
    assert(start.query_read_id_ == end.query_read_id_ && start.target_read_id_ == end.target_read_id_);

    overlap.query_start_position_in_read_ = min(start.query_position_in_read_, end.query_position_in_read_);
    overlap.query_end_position_in_read_   = max(start.query_position_in_read_, end.query_position_in_read_);
    const bool is_negative_strand         = end.target_position_in_read_ < start.target_position_in_read_;
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

__global__ void backtrace_anchors_to_overlaps(const Anchor* const anchors,
                                              Overlap* const overlaps,
                                              int32_t* const overlap_terminal_anchors,
                                              const float* const scores,
                                              bool* const max_select_mask,
                                              const int32_t* const predecessors,
                                              const int64_t n_anchors,
                                              const float min_score)
{
    const int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride    = blockDim.x * gridDim.x;

    for (int i = thread_id; i < n_anchors; i += stride)
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
            Anchor first_anchor         = anchors[first_index];
            overlap_terminal_anchors[i] = i;
            overlaps[i]                 = create_overlap(first_anchor, final_anchor, num_anchors_in_chain);
        }
        else
        {
            max_select_mask[i]          = false;
            overlap_terminal_anchors[i] = -1;
            overlaps[i]                 = create_overlap(empty_anchor(), empty_anchor(), 1);
        }
    }
}

void allocate_anchor_chains(const device_buffer<Overlap>& overlaps,
                            device_buffer<int32_t>& unrolled_anchor_chains,
                            device_buffer<int32_t>& anchor_chain_starts,
                            int64_t& num_total_anchors,
                            DefaultDeviceAllocator allocator,
                            cudaStream_t cuda_stream)
{
    auto thrust_exec_policy = thrust::cuda::par(allocator).on(cuda_stream);
    thrust::plus<int32_t> sum_op;
    num_total_anchors = thrust::transform_reduce(thrust_exec_policy,
                                                 overlaps.begin(),
                                                 overlaps.end(),
                                                 ConvertOverlapToNumResidues(),
                                                 0,
                                                 sum_op);

    unrolled_anchor_chains.clear_and_resize(num_total_anchors);
    anchor_chain_starts.clear_and_resize(overlaps.size());

    thrust::transform_exclusive_scan(thrust_exec_policy,
                                     overlaps.begin(),
                                     overlaps.end(),
                                     anchor_chain_starts.data(),
                                     ConvertOverlapToNumResidues(),
                                     0,
                                     sum_op);
}

__global__ void output_overlap_chains_by_backtrace(const Overlap* const overlaps,
                                                   const Anchor* const anchors,
                                                   const bool* const select_mask,
                                                   const int32_t* const predecessors,
                                                   const int32_t* const chain_terminators,
                                                   int32_t* const anchor_chains,
                                                   int32_t* const anchor_chain_starts,
                                                   const int32_t num_overlaps,
                                                   const bool check_mask)
{
    const int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride    = blockDim.x * gridDim.x;

    // Processes one overlap per iteration,
    // "i" corresponds to an overlap
    for (int i = thread_id; i < num_overlaps; i += stride)
    {

        if (!check_mask || (check_mask & select_mask[i]))
        {
            int32_t anchor_chain_index = 0;
            // As chaining proceeds backwards (i.e., it's a backtrace),
            // we need to fill the new anchor chain array in in reverse order.
            int32_t index = chain_terminators[i];
            while (index != -1)
            {
                anchor_chains[anchor_chain_starts[i] + (overlaps[i].num_residues_ - anchor_chain_index - 1)] = index;
                index                                                                                        = predecessors[index];
                ++anchor_chain_index;
            }
        }
    }
}

__global__ void output_overlap_chains_by_RLE(const Overlap* const overlaps,
                                             const Anchor* const anchors,
                                             const int32_t* const chain_starts,
                                             const int32_t* const chain_lengths,
                                             int32_t* const anchor_chains,
                                             int32_t* const anchor_chain_starts,
                                             const uint32_t num_overlaps)
{
    const int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride    = blockDim.x * gridDim.x;
    for (uint32_t i = thread_id; i < num_overlaps; i += stride)
    {
        int32_t chain_start  = chain_starts[i];
        int32_t chain_length = chain_lengths[i];
        for (int32_t index = chain_start; index < chain_start + chain_length; ++index)
        {
            anchor_chains[index] = index;
        }
    }
}

std::vector<std::string> tokenize_line(std::string line, const char delim = '\t')
{
    std::vector<std::string> tokens;
    std::stringstream st(line);
    std::string tmp;
    while (std::getline(st, tmp, delim))
    {
        tokens.push_back(tmp);
    }
    return tokens;
}

std::vector<seed_debug_entry> read_minimap2_seed_chains(char* seed_file_name)
{

    std::vector<seed_debug_entry> seeds;
    int32_t q_id = 0;

    std::ifstream seed_file_stream;
    seed_file_stream.open(seed_file_name, std::ios::in);
    std::string line;
    if (seed_file_stream.is_open() && seed_file_stream.good())
    {
        std::vector<std::string> current_chain;
        while (std::getline(seed_file_stream, line))
        {
            if (line[0] == '[')
            {
                continue;
            }
            std::vector<std::string> tokens = tokenize_line(line, '\t');
            if (tokens[0] == "QR")
            {
                // Create a new seed_debug_entry
                seed_debug_entry current;
                current.query_id     = tokens[1];
                current.query_int_id = q_id++;
                // Skip the next RS line, as we can't do anything with it yet.
                std::getline(seed_file_stream, line);
                // Retrieve the first seed line
                std::getline(seed_file_stream, line);
                tokens = tokenize_line(line, '\t');
                // Parse all the seeds
                while (tokens[0] == "SD")
                {
                    // process seed line to generate and store a corresponding Anchor
                    seeds.add_seed(tokens);
                    std::getline(seed_file_stream, line);
                    tokens = tokenize_line(line, '\t');
                }
                while (tokens[0] == "CN")
                {
                    // process the chain line and store it
                    seeds.add_chain_entry(tokens);
                    std::getline(seed_file_stream, line);
                    tokens = tokenize_line(line, '\t');
                }
            }
            getline(seed_file_stream, line);
        }

        // Clear the current chain
    }

    seed_file_stream.close();
    return seeds;
}

} // namespace chainerutils
} // namespace cudamapper
} // namespace genomeworks
} // namespace claraparabricks
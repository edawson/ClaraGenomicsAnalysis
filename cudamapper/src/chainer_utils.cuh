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

#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>

// Needed for accumulate - remove when ported to cuda
#include <numeric>
#include <limits>

#include <claraparabricks/genomeworks/cudamapper/types.hpp>
#include <claraparabricks/genomeworks/utils/device_buffer.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{
namespace chainerutils
{

///
/// \brief Stores a chain reported by minimap2's --print-seeds debug mode.
///
struct seed_debug_chain_link
{
    int32_t chain_id;
    int32_t target_id;
    std::vector<Anchor> anchors;
    std::vector<bool> anchor_strands;
};

///
/// \brief Stores the seeds and chains reported by minimap2's --print-seeds debug mode.
///
struct seed_debug_entry
{
    std::string query_id;
    int32_t query_int_id;
    std::unordered_map<std::string, int32_t> target_id_to_int_id;
    int32_t target_id_idx = 0;
    std::vector<Anchor> seeds;
    std::vector<int32_t> seed_lengths;
    std::unordered_map<int32_t, seed_debug_chain_link> chains;

    void add_seed(const std::vector<std::string>& tokens)
    {
        if (target_id_to_int_id.find(tokens[1]) == target_id_to_int_id.end())
        {
            target_id_to_int_id[tokens[1]] = target_id_idx++;
        }
        Anchor a;
        a.query_read_id_           = query_int_id;
        a.target_read_id_          = target_id_to_int_id[tokens[1]];
        a.query_position_in_read_  = std::stoull(tokens[4]); // TODO: verify that the indices are correct here
        a.target_position_in_read_ = std::stoull(tokens[2]);
        seeds.push_back(a);
        seed_lengths.push_back(std::stoull(tokens[5]));
    }
    void add_chain_entry(const std::vector<std::string>& tokens)
    {
        int32_t chain_id = stoll(tokens[1]);
        seed_debug_chain_link link;
        if (chains.find(chain_id) == end(chains))
        {
            chains[chain_id] = link;
        }
        link = chains[chain_id];

        link.chain_id  = chain_id;
        link.target_id = target_id_to_int_id[tokens[2]];
        Anchor a;
        a.query_read_id_           = query_int_id;
        a.target_read_id_          = link.target_id;
        a.query_position_in_read_  = std::stoull(tokens[5]);
        a.target_position_in_read_ = std::stoull(tokens[3]);
        link.anchors.push_back(a);
        link.anchor_strands.push_back(tokens[4] == "+");
    }
};

/// \brief Create an overlap from the first and last anchor in the chain and
/// the total number of anchors in the chain.
///
/// \param start The first anchor in the chain.
/// \param end The last anchor in the chain.
/// \param num_anchors The total number of anchors in the chain.
__host__ __device__ Overlap create_overlap(const Anchor& start,
                                           const Anchor& end,
                                           const int32_t num_anchors);

/// \brief Given an array of anchors and an array denoting the immediate
/// predecessor of each anchor, transform chains of anchors into overlaps.
///
/// \param anchors An array of anchors.
/// \param overlaps An array of overlaps to be filled.
/// \param scores An array of scores. Only chains with a score greater than min_score will be backtraced.
/// \param max_select_mask A boolean mask, used to mask out any chains which are completely contained within larger chains during the backtrace.
/// \param predecessors An array of indices into the anchors array marking the predecessor of each anchor within a chain.
/// \param n_anchors The number of anchors.
/// \param min_score The minimum score of a chain for performing backtracing.
/// Launch configuration: any number of blocks/threads, no dynamic shared memory, cuda_stream must be the same in which anchors/overlaps/scores/mask/predecessors were allocated.
/// example: backtrace_anchors_to_overlaps<<<2048, 32, 0, cuda_stream>>>(...)
__global__ void backtrace_anchors_to_overlaps(const Anchor* const anchors,
                                              Overlap* const overlaps,
                                              int32_t* const overlap_terminal_anchors,
                                              const float* const scores,
                                              bool* const max_select_mask,
                                              const int32_t* const predecessors,
                                              const int64_t n_anchors,
                                              const float min_score);

/// \brief Allocate a 1-dimensional array representing an unrolled ragged array
/// of anchors within each overlap. The final array holds the indices within the anchors array used in chaining
/// of the anchors in the chain.
///
/// \param overlaps An array of Overlaps. Must have a well-formed num_residues_ field
/// \param unrolled_anchor_chains An array of int32_t. Will be resized on return.
/// \param anchor_chain_starts An array holding the index of the first anchor in an overlap within the anchors array used during chaining.
/// \param num_overlaps The number of overlaps in the overlaps array.
/// \param num_total_anchors The total number of anchors across all overlaps.
/// \param allocator The DefaultDeviceAllocator for this overlapper.
/// \param cuda_stream The cudastream to allocate memory within.
void allocate_anchor_chains(const device_buffer<Overlap>& overlaps,
                            device_buffer<int32_t>& unrolled_anchor_chains,
                            device_buffer<int32_t>& anchor_chain_starts,
                            int64_t& num_total_anchors,
                            DefaultDeviceAllocator allocator,
                            cudaStream_t cuda_stream = 0);

/// \brief Given an array of overlaps, fill a 1D unrolled ragged array
/// containing the anchors used to generate each overlap. Anchors must have
/// been chained with a chaining function that fills the predecessors array
/// with the immediate predecessor of each anchor.
///
/// \param overlaps An array of overlaps.
/// \param anchors The array of anchors used to generate overlaps.
/// \param select_mask An array of bools, used to mask overlaps from output.
/// \param predecessors The predecessors array from anchor chaining.
/// \param anchor_chains An array (allocated by allocate_anchor_chains) which will hold the indices of anchors within each chain.
/// \param anchor_chain_starts An array which holds the indices of the first anchor for each overlap in the overlaps array.
/// \param num_overlaps The number of overlaps in the overlaps array
/// \param check_mask A boolean. If true, only overlaps where select_mask is true will have their anchor chains calculated.
/// Launch configuration: any number of blocks/threads, no dynamic shared memory, cuda_stream must be the same in which anchors/overlaps/scores/mask/predecessors/chains/starts were allocated.
/// example: backtrace_anchors_to_overlaps<<<2048, 32, 0, cuda_stream>>>(...)
__global__ void output_overlap_chains_by_backtrace(const Overlap* const overlaps,
                                                   const Anchor* const anchors,
                                                   const bool* const select_mask,
                                                   const int32_t* const predecessors,
                                                   const int32_t* const chain_terminators,
                                                   int32_t* const anchor_chains,
                                                   int32_t* const anchor_chain_starts,
                                                   const int32_t num_overlaps,
                                                   const bool check_mask);

/// \brief Fill a 1D unrolled ragged array with the anchors used to produce each overlap.
///
/// \param overlaps An array of overlaps.
/// \param anchors The array of anchors used to generate overlaps.
/// \param chain_starts An array which holds the indices of the first anchor for each overlap in the overlaps array.
/// \param chain_lengths An array which holds the length of each run of anchors, corresponding to the chain_starts array.
/// \param anchor_chains An array (allocated by allocate_anchor_chains) which will hold the indices of anchors within each chain.
/// \param anchor_chain_starts An array which holds the indices of the first anchor for each overlap in the overlaps array.
/// \param num_overlaps The number of overlaps in the overlaps array.
/// Launch configuration: any number of blocks/threads, no dynamic shared memory, cuda_stream must be the same in which anchors/overlaps/scores/mask/predecessors/chains/starts were allocated.
/// example: backtrace_anchors_to_overlaps<<<2048, 32, 0, cuda_stream>>>(...)
__global__ void output_overlap_chains_by_RLE(const Overlap* const overlaps,
                                             const Anchor* const anchors,
                                             const int32_t* const chain_starts,
                                             const int32_t* const chain_lengths,
                                             int32_t* const anchor_chains,
                                             int32_t* const anchor_chain_starts,
                                             const uint32_t num_overlaps);

std::vector<seed_debug_entry> read_minimap2_seed_chains(const char* const seed_file_name);
struct QueryTargetPair
{
    int32_t query_read_id_;
    int32_t target_read_id_;
    __device__ QueryTargetPair() {}
};

struct QueryReadID
{
    int32_t query_read_id_;
    __device__ QueryReadID(){};
};

// takes the anchor and returns the query read id
struct AnchorToQueryReadIDOp
{
    __device__ __forceinline__ QueryReadID operator()(const Anchor& a) const
    {
        QueryReadID query;
        query.query_read_id_ = a.query_read_id_;
        return query;
    }
};

__device__ bool operator==(const QueryTargetPair& a, const QueryTargetPair& b);

struct OverlapToQueryTargetPairOp
{
    __device__ __forceinline__ QueryTargetPair operator()(const Overlap& a) const
    {
        QueryTargetPair p;
        p.query_read_id_  = a.query_read_id_;
        p.target_read_id_ = a.target_read_id_;
        return p;
    }
};

struct AnchorToQueryTargetPairOp
{
    __device__ __forceinline__ QueryTargetPair operator()(const Anchor& a) const
    {
        QueryTargetPair p;
        p.query_read_id_  = a.query_read_id_;
        p.target_read_id_ = a.target_read_id_;
        return p;
    }
};

__device__ __forceinline__ Anchor empty_anchor()
{
    Anchor empty;
    empty.query_read_id_           = UINT32_MAX;
    empty.target_read_id_          = UINT32_MAX;
    empty.query_position_in_read_  = UINT32_MAX;
    empty.target_position_in_read_ = UINT32_MAX;
    return empty;
}

__device__ bool
operator==(const QueryTargetPair& a, const QueryTargetPair& b);

// TODO VI: replace this function usage in get_overlaps in overlapper_minimap.cu
__global__ void backtrace_anchors_to_overlaps_old(const Anchor* anchors,
                                              Overlap* overlaps,
                                              double* scores,
                                              bool* max_select_mask,
                                              int32_t* predecessors,
                                              const int32_t n_anchors,
                                              const int32_t min_score);

__global__ void backtrace_anchors_to_overlaps(const Anchor* const anchors,
                                              Overlap* const overlaps,
                                              int32_t* const overlap_terminal_anchors,
                                              const float* const scores,
                                              bool* const max_select_mask,
                                              const int32_t* const predecessors,
                                              const int64_t n_anchors,
                                              const float min_score);

__global__ void backtrace_anchors_to_overlaps_debug(const Anchor* anchors,
                                                    Overlap* overlaps,
                                                    double* scores,
                                                    bool* max_select_mask,
                                                    int32_t* predecessors,
                                                    const int32_t n_anchors,
                                                    const int32_t min_score);

void backtrace_anchors_to_overlaps_cpu(const Anchor* anchors,
                                       Overlap* overlaps,
                                       double* scores,
                                       bool* max_select_mask,
                                       int32_t* predecessors,
                                       const int32_t n_anchors,
                                       const int32_t min_score);

__global__ void convert_offsets_to_ends(std::int32_t* starts, std::int32_t* lengths, std::int32_t* ends, std::int32_t n_starts);

__global__ void calculate_tile_starts(const std::int32_t* query_starts,
                                      const std::int32_t* tiles_per_query,
                                      std::int32_t* tile_starts,
                                      const int32_t tile_size,
                                      int32_t num_queries,
                                      const std::int32_t* tiles_per_query_up_to_point);

void encode_query_locations_from_anchors(const Anchor* anchors,
                                         int32_t n_anchors,
                                         device_buffer<int32_t>& query_starts,
                                         device_buffer<int32_t>& query_lengths,
                                         device_buffer<int32_t>& query_ends,
                                         int32_t& n_queries,
                                         DefaultDeviceAllocator& _allocator,
                                         cudaStream_t& _cuda_stream);

// void encode_tile_starts_and_ends(device_buffer<int32_t>& starts,
//                                  device_buffer<int32_t>& lengths,
//                                  device_buffer<int32_t>& ends,
//                                  device_buffer<int32_t>& tiles_per_entry,
//                                  device_buffer<int32_t>& tile_starts,
//                                  device_buffer<int32_t>& tile_ends,
//                                  int32_t num_entries,
//                                  int32_t num_tiles,
//                                  DefaultDeviceAllocator& _allocator,
//                                  cudaStream_t& _cuda_stream);

void encode_anchor_query_target_pairs(const Anchor* anchors,
                                      int32_t n_anchors,
                                      int32_t tile_size,
                                      device_buffer<int32_t>& query_target_pair_starts,
                                      device_buffer<int32_t>& query_target_pair_lengths,
                                      device_buffer<int32_t>& query_target_pair_ends,
                                      device_buffer<int32_t>& tiles_per_qt_pair,
                                      int32_t& n_query_target_pairs,
                                      int32_t& n_qt_tiles,
                                      DefaultDeviceAllocator& _allocator,
                                      cudaStream_t& _cuda_stream,
                                      int32_t block_size);

void encode_overlap_query_target_pairs(Overlap* overlaps,
                                       int32_t n_overlaps,
                                       device_buffer<int32_t>& query_target_pair_starts,
                                       device_buffer<int32_t>& query_target_pair_lengths,
                                       device_buffer<int32_t>& query_target_pair_ends,
                                       int32_t& n_query_target_pairs,
                                       DefaultDeviceAllocator& _allocator,
                                       cudaStream_t& _cuda_stream,
                                       int32_t block_size);

__global__ void initialize_mask(bool* anchors_mask, const int32_t n_anchors, bool val);

__global__ void initialize_array(int32_t* array, const int32_t num_values, int32_t value);
__global__ void initialize_array(double* array, const int32_t num_values, double value);

} // namespace chainerutils
} // namespace cudamapper
} // namespace genomeworks
} // namespace claraparabricks
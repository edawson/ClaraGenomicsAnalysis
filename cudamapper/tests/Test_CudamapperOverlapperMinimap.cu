/*
* Copyright 2019-2020 NVIDIA CORPORATION.
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

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../src/chainer_utils.cuh"
#include "../src/overlapper_minimap.hpp"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

TEST(TestCudamapperOverlapperMinimap, Produce_Chains_Small_Chain)
{

    DefaultDeviceAllocator allocator = create_default_device_allocator();
    CudaStream cuda_stream           = make_cuda_stream();
    int32_t num_anchors              = 10;

    std::vector<Overlap> overlaps(num_anchors);
    std::vector<int32_t> predecessors{-1, 0, 1, -1, 2, 3, 5, -1, 4, 8};
    std::vector<Anchor> anchors(num_anchors);
    Anchor blank;
    blank.query_read_id_           = 0;
    blank.target_read_id_          = 1;
    blank.query_position_in_read_  = 10;
    blank.target_position_in_read_ = 100;
    for (int i = 0; i < num_anchors; ++i)
    {
        anchors[i] = blank;
    }
    std::vector<double> scores{0, 12, 20, 0, 20, 20, 20, 0, 20, 20};

    device_buffer<Overlap> d_overlaps(num_anchors, allocator, cuda_stream.get());
    device_buffer<int32_t> d_predecessors(num_anchors, allocator, cuda_stream.get());
    device_buffer<bool> d_mask(num_anchors, allocator, cuda_stream.get());
    device_buffer<Anchor> d_anchors(num_anchors, allocator, cuda_stream.get());
    device_buffer<double> d_scores(num_anchors, allocator, cuda_stream.get());

    cudautils::device_copy_n(overlaps.data(), num_anchors, d_overlaps.data(), cuda_stream.get());
    cudautils::device_copy_n(predecessors.data(), num_anchors, d_predecessors.data(), cuda_stream.get());
    cudautils::device_copy_n(anchors.data(), num_anchors, d_anchors.data(), cuda_stream.get());
    cudautils::device_copy_n(scores.data(), num_anchors, d_scores.data(), cuda_stream.get());

    chainerutils::chain_anchors_by_backtrace<<<1024, 64, 0, cuda_stream.get()>>>(d_anchors.data(),
                                                                                 d_overlaps.data(),
                                                                                 d_scores.data(),
                                                                                 d_mask.data(),
                                                                                 d_predecessors.data(),
                                                                                 num_anchors,
                                                                                 15);

    cudautils::device_copy_n(d_overlaps.data(), num_anchors, overlaps.data(), cuda_stream.get());
    ASSERT_EQ(overlaps[9].query_start_position_in_read_, 10);
    ASSERT_EQ(overlaps[9].num_residues_, 6);
    ASSERT_EQ(overlaps[0].num_residues_, 0);
    ASSERT_EQ(overlaps[1].num_residues_, 0);
    ASSERT_EQ(overlaps[2].num_residues_, 3);

    auto cu_ptr = cuda_stream.get();

    int32_t num_maxes = chainerutils::count_unmasked(const_cast<const bool*>(d_mask.data()), num_anchors, allocator, cu_ptr);
    ASSERT_EQ(num_maxes, 2);
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks

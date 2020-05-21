/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"
#include <string>
#include <vector>
#include "../src/cudamapper_utils.cpp"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

TEST(MMH3, hashes_chars)
{
    std::string s("ACTGGCTTGC");
    char* c             = const_cast<char*>(s.c_str());
    std::uint32_t* hval = new uint32_t[1];
    MurmurHash3_x86_32(c, 4, 42, hval);
    std::cerr << *hval << std::endl;
}

TEST(SimilarityTest, similarity_of_identical_seqs_is_1)
{
    std::string a("AAACCTATGAGGG");
    std::string b("AAACCTATGAGGG");
    std::string long_b("AAACCTATGAGGGAAACCTATGAGGG");

    char* raw_a = const_cast<char*>(a.c_str());
    // float sim   = fast_sequence_similarity();
    // ASSERT_EQ(sim, 1.0);
}
// TEST(SimilarityTest, similarity_of_disjoint_seqs_is_0)
// {
//     std::string a("AAACCTATGAGGG");
//     std::string b("CCCAATTTAAATT");
//     float sim = sequence_jaccard_similarity(a, b, 4, 1);
//     ASSERT_EQ(sim, 0.0);
// }
// TEST(SimilarityTest, similarity_of_similar_seqs_is_accurate_estimate)
// {
//     std::string a("AAACCTATGAGGG");
//     std::string b("AAACCTAAGAGGG");
//     float sim = sequence_jaccard_similarity(a, b, 4, 1);
//     ASSERT_GT(sim, 0.0);
//     ASSERT_LT(sim, 1.0);
// }
} // namespace cudamapper

} // namespace genomeworks
} // namespace claraparabricks
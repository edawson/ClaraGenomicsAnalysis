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

TEST(MurmurHash, murmurhash_values_are_same_or_different_as_expect)
{
    std::string s("ACTGGCTTGC");
    std::string t("ACTGGCTTGC");
    char* c               = const_cast<char*>(s.c_str());
    char* d               = const_cast<char*>(t.c_str());
    std::uint32_t* hval_c = new uint32_t;
    std::uint32_t* hval_d = new uint32_t;
    MurmurHash3_x86_32(c, 4, 42, hval_c);
    MurmurHash3_x86_32(d, 4, 42, hval_d);
    ASSERT_EQ(*hval_c, *hval_d);

    MurmurHash3_x86_32(c + 4, 5, 42, hval_c);
    MurmurHash3_x86_32(d + 4, 5, 42, hval_d);
    ASSERT_EQ(*hval_c, *hval_d);

    MurmurHash3_x86_32(c + 1, 9, 42, hval_c);
    MurmurHash3_x86_32(d + 3, 4, 42, hval_d);
    ASSERT_NE(*hval_c, *hval_d);

    delete hval_c;
    delete hval_d;
}

TEST(SimilarityTest, similarity_of_identical_seqs_is_1)
{
    std::string a("AAACCTATGAGGG");
    std::string b("AAACCTATGAGGG");
    std::string long_b("AAACCTATGAGGGAAACCTATGAGGG");

    char* raw_a                = const_cast<char*>(a.c_str());
    char* raw_b                = const_cast<char*>(b.c_str());
    std::uint32_t arr_size     = 1000;
    std::uint32_t* count_array = new std::uint32_t[arr_size];
    float sim                  = fast_sequence_similarity(raw_a, 0, 13, raw_b, 0, 13, 5, arr_size, count_array, false);
    ASSERT_EQ(sim, 1.0);
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
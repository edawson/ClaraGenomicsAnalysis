/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <mutex>
#include <vector>

#include <claragenomics/cudamapper/types.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace io
{
class FastaParser;
}; // namespace io

namespace cudamapper
{

/// \brief prints overlaps to stdout in <a href="https://github.com/lh3/miniasm/blob/master/PAF.md">PAF format</a>
/// \param overlaps vector of overlap objects
/// \param cigar cigar strings
/// \param query_parser needed for read names and lenghts
/// \param target_parser needed for read names and lenghts
/// \param kmer_size minimizer kmer size
/// \param write_output_mutex mutex that enables exclusive access to output stream
/// \param number_of_devices function uses hardware_concurrency()/number_of_devices threads
void print_paf(const std::vector<Overlap>& overlaps,
               const std::vector<std::string>& cigar,
               const io::FastaParser& query_parser,
               const io::FastaParser& target_parser,
               int32_t kmer_size,
               std::mutex& write_output_mutex,
               int32_t number_of_devices);

///
/// \brief Calculate the jaccard sequence similarity of the sequences a[a_start:a_end] and b[b_start:b_end]
/// using the Jaccard similarity coefficient of the kmer spaces of the sequences.
/// \param a A C string DNA sequence.
/// \param a_start The start position on a.
/// \param a_end The end position on a.
/// \param b A C string DNA sequence.
/// \param b_start The start position on b.
/// \param b_end The end position on b.
/// \param kmer_size The kmer size to use for comparison
/// \param array_size The size of the internal array to use for counting. Larger arrays may be more accurate at the expense of memory usage.
/// \param reversed Whether the overlap is reversed. If true, use the reverse-complement of the opposite ends of b in calculating similarity.
/// \return A float of the approximate sequence similarity
float fast_sequence_similarity(char*& a,
                               position_in_read_t a_start,
                               position_in_read_t a_end,
                               char*& b,
                               position_in_read_t b_start,
                               position_in_read_t b_end,
                               std::int32_t kmer_size,
                               std::int32_t array_size,
                               uint32_t*& count_array,
                               bool reversed);

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks

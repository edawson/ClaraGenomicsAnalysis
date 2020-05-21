/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <algorithm>
#include <cassert>
#include <thread>
#include <vector>
#include <cstring>

#include "cudamapper_utils.hpp"

#include <claragenomics/io/fasta_parser.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>

#include <iostream>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

namespace
{

/// \brief prints part of data passed to print_paf(), to be run in a separate thread
/// \param overlaps see print_paf()
/// \param cigar see print_paf()
/// \param query_parser see print_paf()
/// \param target_parser see print_paf()
/// \param target_parser see print_paf()
/// \param kmer_size see print_paf()
/// \param write_output_mutex see print_paf()
/// \param first_overlap_to_print index of first overlap from overlaps to print
/// \param number_of_overlaps_to_print number of overlaps to print
void print_part_of_data(const std::vector<Overlap>& overlaps,
                        const std::vector<std::string>& cigar,
                        const io::FastaParser& query_parser,
                        const io::FastaParser& target_parser,
                        const int32_t kmer_size,
                        std::mutex& write_output_mutex,
                        const int64_t first_overlap_to_print,
                        const int64_t number_of_overlaps_to_print)
{
    if (number_of_overlaps_to_print <= 0)
    {
        return;
    }

    // All overlaps are saved to a single vector of chars and that vector is then printed to output.
    // Writing overlaps directly to output would be inefficinet as all writes to output have to protected by a mutex.
    //
    // Allocate approximately 150 characters for each overlap which will be processed by this thread,
    // if more characters are needed buffer will be reallocated.
    std::vector<char> buffer(150 * number_of_overlaps_to_print);
    // characters written buffer so far
    int64_t chars_in_buffer = 0;

    for (int64_t i = first_overlap_to_print; i < first_overlap_to_print + number_of_overlaps_to_print; ++i)
    {
        const std::string& query_read_name  = query_parser.get_sequence_by_id(overlaps[i].query_read_id_).name;
        const std::string& target_read_name = target_parser.get_sequence_by_id(overlaps[i].target_read_id_).name;
        // (over)estimate the number of character that are going to be needed
        // 150 is an overestimate of number of characters that are going to be needed for non-string values
        int32_t expected_chars = 150 + get_size<int32_t>(query_read_name) + get_size<int32_t>(target_read_name);
        if (!cigar.empty())
        {
            expected_chars += get_size<int32_t>(cigar[i]);
        }
        // if there is not enough space in buffer reallocate
        if (get_size<int64_t>(buffer) - chars_in_buffer < expected_chars)
        {
            buffer.resize(buffer.size() * 2 + expected_chars);
        }
        // Add basic overlap information.
        const int32_t added_chars = std::sprintf(buffer.data() + chars_in_buffer,
                                                 "%s\t%lu\t%i\t%i\t%c\t%s\t%lu\t%i\t%i\t%i\t%ld\t%i",
                                                 query_read_name.c_str(),
                                                 query_parser.get_sequence_by_id(overlaps[i].query_read_id_).seq.length(),
                                                 overlaps[i].query_start_position_in_read_,
                                                 overlaps[i].query_end_position_in_read_,
                                                 static_cast<unsigned char>(overlaps[i].relative_strand),
                                                 target_read_name.c_str(),
                                                 target_parser.get_sequence_by_id(overlaps[i].target_read_id_).seq.length(),
                                                 overlaps[i].target_start_position_in_read_,
                                                 overlaps[i].target_end_position_in_read_,
                                                 overlaps[i].num_residues_ * kmer_size, // Print out the number of residue matches multiplied by kmer size to get approximate number of matching bases
                                                 std::max(std::abs(static_cast<std::int64_t>(overlaps[i].target_start_position_in_read_) - static_cast<std::int64_t>(overlaps[i].target_end_position_in_read_)),
                                                          std::abs(static_cast<std::int64_t>(overlaps[i].query_start_position_in_read_) - static_cast<std::int64_t>(overlaps[i].query_end_position_in_read_))), //Approximate alignment length
                                                 255);
        chars_in_buffer += added_chars;
        // If CIGAR string is generated, output in PAF.
        if (!cigar.empty())
        {
            const int32_t added_cigar_chars = std::sprintf(buffer.data() + chars_in_buffer,
                                                           "\tcg:Z:%s",
                                                           cigar[i].c_str());
            chars_in_buffer += added_cigar_chars;
        }
        // Add new line to demarcate new entry.
        buffer[chars_in_buffer] = '\n';
        ++chars_in_buffer;
    }
    buffer[chars_in_buffer] = '\0';

    std::lock_guard<std::mutex> lg(write_output_mutex);
    printf("%s", buffer.data());
}

} // namespace

void print_paf(const std::vector<Overlap>& overlaps,
               const std::vector<std::string>& cigar,
               const io::FastaParser& query_parser,
               const io::FastaParser& target_parser,
               const int32_t kmer_size,
               std::mutex& write_output_mutex,
               const int32_t number_of_devices)
{
    assert(!cigar.empty() || (overlaps.size() == cigar.size()));

    // divide the work into several threads

    int32_t number_of_threads   = std::thread::hardware_concurrency() / number_of_devices; // We could use a better heuristic here
    int64_t overlaps_per_thread = get_size<int64_t>(overlaps) / number_of_threads;

    std::vector<std::thread> threads;

    for (int32_t thread_id = 0; thread_id < number_of_threads; ++thread_id)
    {
        threads.emplace_back(print_part_of_data,
                             std::ref(overlaps),
                             std::ref(cigar),
                             std::ref(query_parser),
                             std::ref(target_parser),
                             kmer_size,
                             std::ref(write_output_mutex),
                             thread_id * overlaps_per_thread,
                             thread_id != number_of_threads - 1 ? overlaps_per_thread : get_size<int64_t>(overlaps) - thread_id * overlaps_per_thread); // last thread prints all remaining overlaps
    }

    for (std::thread& thread : threads)
    {
        thread.join();
    }
}

std::uint32_t MurmurHash3_x86_32(void* key, std::int32_t length, std::uint32_t seed, std::uint32_t* hash_value)
{
    const uint8_t* data             = reinterpret_cast<const uint8_t*>(key);
    const std::uint32_t block_count = length / 4; // 4 8-bit "blocks" in a 32-bit int

    std::uint32_t hash_start  = seed;
    const uint32_t constant_1 = 0xcc9e2d51;
    const uint32_t constant_2 = 0x1b873593;
    const uint32_t* blocks    = reinterpret_cast<const uint32_t*>(data + block_count * 4);

    for (std::int32_t i = -block_count; i; i++)
    {

        uint32_t k = blocks[i];

        k *= constant_1;
        k = ((k << 15) | (k >> (32 - 15)));
        k *= constant_2;
        hash_start ^= k;
        hash_start = ((hash_start << 15) | (hash_start >> (32 - 13)));
        hash_start = hash_start * 5 + 0xe6546b64;
    }

    const uint8_t* tail = reinterpret_cast<const uint8_t*>(data + block_count * 4);

    std::uint32_t k = 0;
    switch (length & 3)
    {
    case 3: k ^= tail[2] << 16;
    case 2: k ^= tail[1] << 8;
    case 1:
        k ^= tail[0];
        k *= constant_1;
        k = ((k << 15) | (k >> (32 - 15)));
        k *= constant_2;
        hash_start ^= k;
    };

    hash_start ^= length;

    hash_start ^= hash_start >> 16;
    hash_start *= 0x85ebca6b;
    hash_start ^= hash_start >> 13;
    hash_start *= 0xc2b2ae35;
    hash_start ^= hash_start >> 16;

    *hash_value = hash_start;
}

float fast_sequence_similarity(char*& a,
                               position_in_read_t a_start,
                               position_in_read_t a_end,
                               char*& b,
                               position_in_read_t b_start,
                               position_in_read_t b_end,
                               std::int32_t kmer_size,
                               std::int32_t array_size,
                               bool reversed)
{
    std::uint32_t intersection_size = 0;

    const position_in_read_t a_length = a_end - a_start;
    const std::uint32_t num_a_kmers   = a_length - kmer_size + 1;
    const position_in_read_t b_length = b_end - b_start;
    const std::uint32_t num_b_kmers   = b_length - kmer_size + 1;

    std::uint32_t* count_array = new std::uint32_t[array_size];
    memset(count_array, (uint32_t)0, array_size * sizeof(count_array[0]));

    std::uint32_t* kmer_hash = new std::uint32_t[1];

    for (std::size_t i = 0; i < num_a_kmers; ++i)
    {
        MurmurHash3_x86_32(a + i, kmer_size, 42, kmer_hash);
        count_array[(*kmer_hash) % array_size] = 1;
    }

    for (std::size_t i = 0; i < num_b_kmers; ++i)
    {
        MurmurHash3_x86_32(b + i, kmer_size, 42, kmer_hash);
        if (count_array[(*kmer_hash) % array_size] == 1)
        {
            ++intersection_size;
        }
    }

    delete[] kmer_hash;

    const std::uint32_t union_size = num_a_kmers + num_b_kmers - intersection_size;
    delete[] count_array;

    float similarity = static_cast<float>(intersection_size) / static_cast<float>(union_size);
    return similarity;
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks

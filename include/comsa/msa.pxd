from libc.stdint cimport uint8_t, uint32_t

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string


cdef extern from "msa.h" nogil:

    cppclass CMSACompress:
        CMSACompress()

        bool CompressFasta "Compress"(
            vector[string] &v_names, 
            vector[string] &v_sequences, 
            vector[uint8_t] &compressed_data, 
		    size_t &comp_text_size, 
            size_t &comp_seq_size, 
            bool _fast_variant
        ) except +
        bool CompressStockholm "Compress"(
            vector[vector[uint8_t]] &v_meta, 
            vector[uint32_t] &v_offsets, 
            vector[string] &v_names, 
            vector[string] &v_sequences, 
            vector[uint8_t] &v_compressed_data,
            size_t &comp_text_size, 
            size_t &comp_seq_size, 
            bool _fast_variant
        ) except +

        bool DecompressFasta "Decompress"(
            vector[uint8_t] &v_compressed_data, 
            vector[string] &v_names, 
            vector[string] &v_sequences
        ) except +
        bool DecompressStockholm "Decompress"(
            vector[uint8_t] &v_compressed_data, 
            vector[vector[uint8_t]] &v_meta, 
            vector[uint32_t] &v_offsets, 
            vector[string] &v_names, 
            vector[string] &v_sequences
        ) except +

from libcpp.string cimport string


cdef extern from "defs.h" nogil:

    struct stockholm_family_desc_t:
        size_t n_sequences
        size_t n_columns
        size_t raw_size
        size_t compressed_size
        size_t compressed_data_ptr
        string ID
        string AC
        
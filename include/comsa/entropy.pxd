cdef extern from "entropy.h" namespace "ctx_length_t" nogil:

    cppclass ctx_length_t:
        pass

cdef extern from "entropy.h" namespace "ctx_length_t" nogil:

    const ctx_length_t tiny
    const ctx_length_t small
    const ctx_length_t medium
    const ctx_length_t large
    const ctx_length_t huge

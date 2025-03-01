# coding: utf-8
# cython: language_level=3, linetrace=True, binding=True

from libcpp cimport bool
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp.string cimport string

from cpython.buffer cimport PyBUF_READ, PyBUF_WRITE
from cpython.memoryview cimport PyMemoryView_FromMemory

from comsa.msa cimport CMSACompress
from comsa.defs cimport stockholm_family_desc_t


import io
import os
import struct

__version__ = PROJECT_VERSION


def decompress(object data):

    cdef CMSACompress compressor
    cdef const uint8_t[::1] view = data

    cdef vector[string] vnames
    cdef vector[string] vseqs
    cdef vector[uint8_t] vdata

    for i in range(view.shape[0]):
        vdata.push_back(view[i])

    compressor.Decompress(
        vdata,
        vnames,
        vseqs
    )

    return list(vnames), list(vseqs)


cdef class MSA:
    cdef readonly string         id
    cdef readonly string         accession
    cdef readonly vector[string] names
    cdef readonly vector[string] sequences
    cdef          vector[string] meta

    def __init__(self, id, accession, names, sequences):
        self.names = names
        self.sequences = sequences


cdef class MSACReader:

    cdef vector[stockholm_family_desc_t] families
    cdef dict                            index
    cdef object                          file
    cdef str                             size_format
    cdef int                             size_size

    def __init__(self, object file, str size_format = 'N'):
        self.size_format = size_format
        self.size_size = struct.calcsize(size_format)
        self.file = io.BufferedReader(file)
        self._preload()

    def _preload(self):
        cdef uint64_t footer_size
        cdef uint64_t logical_file_size

        logical_file_size = self.file.seek(-self.size_size, os.SEEK_END)
        footer_size = self._load_uint(True)
        
        if footer_size > logical_file_size:
            raise ValueError("Failed to parse footer size, file may be corrupted")

        self.file.seek(-(<int> self.size_size + <int> footer_size), os.SEEK_END)
        self._preload_family_descriptions(logical_file_size)

    cdef uint64_t _load_uint(self, bool fixed_size = False):
        cdef uint32_t  shift   = 0
        cdef uint32_t  n_bytes = self.size_size
        cdef size_t    x       = 0
        cdef bytearray buffer  = bytearray(1)

        if not fixed_size:
            if not self.file.readinto(buffer):
                raise EOFError(self.file.tell())
            n_bytes = struct.unpack('B', buffer)[0]

        for i in range(n_bytes):
            if not self.file.readinto(buffer):
                raise EOFError(self.file.tell())
            x += struct.unpack('B', buffer)[0] << shift
            shift += 8

        return x

    cdef stockholm_family_desc_t _load_family_desc(self):
        cdef stockholm_family_desc_t fd
        
        fd.n_sequences = self._load_uint()
        fd.n_columns = self._load_uint()
        fd.raw_size = self._load_uint()
        fd.compressed_size = self._load_uint()
        fd.compressed_data_ptr = self._load_uint()
        fd.ID.clear()
        for c in iter(lambda: self.file.read(1), b'\0'):
            fd.ID.push_back(ord(c))
        fd.AC.clear()
        for c in iter(lambda: self.file.read(1), b'\0'):
            fd.AC.push_back(ord(c))
        
        return fd

    def _preload_family_descriptions(self, int logical_file_size):
        cdef stockholm_family_desc_t fd
        cdef bytearray               buffer = bytearray(1)

        self.families.clear()
        while self.file.tell() < logical_file_size:
            fd = self._load_family_desc()
            self.families.push_back(fd)

        self.index = {
            self.families[i].ID.decode():i
            for i in range(self.families.size())
        }

    def __len__(self):
        return self.families.size()

    def __getitem__(self, object key):
        cdef CMSACompress            comp
        cdef size_t                  index  = self.index[key]
        cdef size_t                  offset = self.families[index].compressed_data_ptr
        cdef vector[uint8_t]         data
        cdef vector[vector[uint8_t]] meta
        cdef vector[uint32_t]        offsets
        cdef memoryview              mview
        cdef MSA                     msa

        # NB: for some reason this here doesn't use `load_uint` 
        #     in the original code, which makes it non-portable
        #     i guess?
        self.file.seek(offset, os.SEEK_SET)
        length = struct.unpack(self.size_format, self.file.read(self.size_size))[0]    

        data.resize(length)
        mview = PyMemoryView_FromMemory(<char*> data.data(), length, PyBUF_WRITE)
        self.file.readinto(mview)
        
        msa = MSA.__new__(MSA)
        msa.id = self.families[index].ID
        msa.accession = self.families[index].AC
        comp.Decompress(data, meta, offsets, msa.names, msa.sequences)
        return msa

    def __iter__(self):
        return map(self.__getitem__, iter(self.keys()))

    def keys(self):
        return self.index.keys()






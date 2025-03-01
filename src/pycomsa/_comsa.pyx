# coding: utf-8
# cython: language_level=3, linetrace=True, binding=True

# --- C imports ----------------------------------------------------------------

from libcpp cimport bool
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp.string cimport string

from cpython.buffer cimport PyBUF_READ, PyBUF_WRITE
from cpython.memoryview cimport PyMemoryView_FromMemory
from cpython.pythread cimport (
    PyThread_type_lock,
    PyThread_allocate_lock,
    PyThread_free_lock,
    PyThread_acquire_lock,
    PyThread_release_lock,
    WAIT_LOCK,
)

cimport comsa.entropy
from comsa.msa cimport CMSACompress
from comsa.defs cimport stockholm_family_desc_t

# --- Python imports -----------------------------------------------------------

import builtins
import collections
import contextlib
import io
import os
import struct

__version__ = PROJECT_VERSION

# --- Classes ------------------------------------------------------------------

cdef class MSA:
    cdef readonly string         id
    cdef readonly string         accession
    cdef readonly vector[string] names
    cdef readonly vector[string] sequences
    cdef          vector[string] meta

    def __init__(self, id, accession, names, sequences):
        self.names = names
        self.sequences = sequences


cdef class FileGuard:
    """A mutex wrapping a file to avoid concurrent accesses.
    """

    cdef object             file
    cdef PyThread_type_lock lock

    def __cinit__(self):
        self.lock = PyThread_allocate_lock()

    def __init__(self, object file):
        self.file = file

    def __del__(self):
        PyThread_free_lock(self.lock)

    def __enter__(self):
        PyThread_acquire_lock(self.lock, WAIT_LOCK)
        return self.file

    def __exit__(self, *exc_details):
        PyThread_release_lock(self.lock)


cdef class _StockholmReader:

    cdef vector[stockholm_family_desc_t] families
    cdef dict                            index
    cdef FileGuard                       guard
    cdef str                             size_format
    cdef int                             size_size
    cdef size_t                          length
    cdef vector[uint8_t]                 data

    def __init__(self, object file, str size_format = 'N'):
        self.size_format = size_format
        self.size_size = struct.calcsize(size_format)
        self.guard = FileGuard(io.BufferedReader(file))

        with self.guard as file:
            self.length = file.seek(0, os.SEEK_END)
            self._preload(file)

    def _preload(self, file):
        cdef uint64_t footer_size
        cdef uint64_t logical_file_size

        logical_file_size = file.seek(-self.size_size, os.SEEK_END)
        footer_size = self._load_uint(file, fixed_size=True)

        if footer_size > logical_file_size:
            raise ValueError("Failed to parse footer size, file may be corrupted")

        file.seek(-(<int> self.size_size + <int> footer_size), os.SEEK_END)
        self._preload_family_descriptions(file, logical_file_size)

    cdef uint64_t _load_uint(self, object file, bool fixed_size = False):
        cdef uint32_t  shift   = 0
        cdef uint32_t  n_bytes = self.size_size
        cdef size_t    x       = 0
        cdef bytearray buffer  = bytearray(1)

        if not fixed_size:
            if not file.readinto(buffer):
                raise EOFError(f"Failed to load integer")
            n_bytes = struct.unpack('B', buffer)[0]

        for i in range(n_bytes):
            if not file.readinto(buffer):
                raise EOFError(f"Failed to integer")
            x += struct.unpack('B', buffer)[0] << shift
            shift += 8

        return x

    cdef stockholm_family_desc_t _load_family_desc(self, object file):
        cdef stockholm_family_desc_t fd

        fd.n_sequences = self._load_uint(file)
        fd.n_columns = self._load_uint(file)
        fd.raw_size = self._load_uint(file)
        fd.compressed_size = self._load_uint(file)
        fd.compressed_data_ptr = self._load_uint(file)
        fd.ID.clear()
        for c in iter(lambda: file.read(1), b'\0'):
            fd.ID.push_back(ord(c))
        fd.AC.clear()
        for c in iter(lambda: file.read(1), b'\0'):
            fd.AC.push_back(ord(c))

        return fd

    cdef void _preload_family_descriptions(self, object file, int logical_file_size):
        cdef stockholm_family_desc_t fd
        cdef bytearray               buffer = bytearray(1)

        self.families.clear()
        while file.tell() < logical_file_size:
            fd = self._load_family_desc(file)
            self.families.push_back(fd)

        self.index = {
            self.families[i].ID.decode():i
            for i in range(self.families.size())
        }

    def __len__(self):
        return self.families.size()

    def __getitem__(self, object key):
        cdef size_t index = self.index[key]
        return self.family(index)

    cpdef MSA family(self, size_t index):
        cdef CMSACompress            comp
        cdef size_t                  offset = self.families[index].compressed_data_ptr
        cdef vector[vector[uint8_t]] meta
        cdef vector[uint32_t]        offsets
        cdef memoryview              mview
        cdef MSA                     msa

        with self.guard as file:
            # NB: for some reason this here doesn't use `load_uint`
            #     in the original code, which makes it non-portable
            #     i guess?
            file.seek(offset, os.SEEK_SET)
            length = struct.unpack(self.size_format, file.read(self.size_size))[0]
            self.data.resize(length)
            mview = PyMemoryView_FromMemory(<char*> self.data.data(), length, PyBUF_WRITE)
            if file.readinto(mview) != length:
                raise EOFError()

        if not _is_context_byte(self.data[0]):
            raise ValueError(f"Invalid context byte at offset {offset + self.size_size}: {chr(self.data[0])!r}")

        msa = MSA.__new__(MSA)
        msa.id = self.families[index].ID
        msa.accession = self.families[index].AC

        try:
            with nogil:
                comp.DecompressStockholm(self.data, meta, offsets, msa.names, msa.sequences)
        except Exception as e:
            raise ValueError("Failed to decompress data") from e

        return msa

    def keys(self):
        return self.index.keys()


cdef class _FastaReader:

    cdef FileGuard                       guard
    cdef size_t                          length
    cdef vector[uint8_t]                 data

    def __init__(self, object file):
        self.guard = FileGuard(io.BufferedReader(file))
        with self.guard as file:
            self.length = file.seek(0, os.SEEK_END)
        self.data.resize(self.length)

    cpdef MSA family(self):
        cdef CMSACompress            comp
        cdef memoryview              mview
        cdef MSA                     msa

        with self.guard as file:
            file.seek(0, os.SEEK_SET)
            mview = PyMemoryView_FromMemory(<char*> self.data.data(), self.length, PyBUF_WRITE)
            if file.readinto(mview) != self.length:
                raise EOFError()

        if not _is_context_byte(self.data[0]):
            raise ValueError(f"Invalid context byte at offset 0: {chr(self.data[0])!r}")

        msa = MSA.__new__(MSA)
        msa.id.clear()
        msa.accession.clear()

        try:
            with nogil:
                comp.DecompressFasta(self.data, msa.names, msa.sequences)
        except Exception as e:
            raise ValueError("Failed to decompress data") from e

        return msa


class StockholmReader(collections.abc.Sequence):
    """A reader of a multi-family CoMSA file.
    """

    def __init__(self, object file, str size_format = "N"):
        self._reader = _StockholmReader(file, size_format=size_format)

    def __len__(self):
        return self._reader.__len__()

    def __getitem__(self, object index):
        cdef ssize_t length = self._reader.__len__()
        cdef ssize_t index_ = index
        if index_ < 0:
            index_ += length
        if index_ < 0 or index_ >= length:
            raise IndexError(index)
        return self._reader.family(index_)


class FastaReader(collections.abc.Sequence):

    def __init__(self, object file):
        self._reader = _FastaReader(file)

    def __len__(self):
        return 1

    def __getitem__(self, object index):
        if index > 0 or index < -1:
            raise IndexError(index)
        return self._reader.family()


# --- Functions ----------------------------------------------------------------

def _is_context_byte(b):
    return b in {
        <int> comsa.entropy.tiny,
        <int> comsa.entropy.small,
        <int> comsa.entropy.medium,
        <int> comsa.entropy.large,
        <int> comsa.entropy.huge,
        64 | <int> comsa.entropy.tiny,
        64 | <int> comsa.entropy.small,
        64 | <int> comsa.entropy.medium,
        64 | <int> comsa.entropy.large,
        64 | <int> comsa.entropy.huge
    }


def _detect_format(file, size_format = "N"):
    """Attempt to detect format of file (FASTA or Stockholm compressed).
    """
    # compute sizeof(size_t) given the provided format
    n = struct.calcsize(size_format)

    # get file length to check the loaded length for the first block
    # is consistent
    length = file.seek(0, os.SEEK_END)
    file.seek(0, os.SEEK_SET)
    peek = file.peek()

    # for a Stockholm file, the file starts with the length of the first
    # block, so the first N byte should encode a valid length, and the
    # byte N+1 should be a context byte
    l = struct.unpack(size_format, peek[:n])[0]
    is_valid_length = l < length
    ctx_stockholm = _is_context_byte(peek[n])
    if ctx_stockholm and is_valid_length:
        return "stockholm"

    # for a FASTA file, the file starts immediately with a compressed block,
    # so the first byte needs to be a context byte
    ctx_fasta = _is_context_byte(peek[0])
    if ctx_fasta:
        return "fasta"

    # if none of these are valid, the file may have been obtained with
    # a different architecture (size_t, endianess), or may just be invalid.
    raise ValueError("Failed to detect format of file")


@contextlib.contextmanager
def open(file, mode = "r", format = "detect", size_format = "N"):

    if not hasattr(file, "read"):
        file = builtins.open(file, "rb")
        close = True
    else:
        file = io.BufferedReader(file)
        close = False

    try:
        if mode == "r":
            if format == "detect":
                format = _detect_format(file, size_format)
            if format == "fasta":
                yield FastaReader(file)
            elif format == "stockholm":
                yield StockholmReader(file)

        elif mode == "w":
            raise NotImplementedError

        else:
            raise ValueError(f"invalid mode: {mode!r}")
    finally:
        if close:
            file.close()
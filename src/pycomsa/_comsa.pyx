# coding: utf-8
# cython: language_level=3, linetrace=True, binding=True

"""Bindings to CoMSA, a compressor for multiple sequence alignments.
"""

# --- C imports ----------------------------------------------------------------

from libcpp cimport bool
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.algorithm cimport copy_n

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

cdef extern from * nogil:
    # https://stackoverflow.com/a/217605
    """
    inline void ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
    }

    inline void rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), s.end());
    }
    """
    void rtrim(string& s)
    void ltrim(string& s)

# --- Python imports -----------------------------------------------------------

import builtins
import collections
import contextlib
import io
import itertools
import os
import struct

__version__ = PROJECT_VERSION

# --- MSA ----------------------------------------------------------------------

cdef class _MSASequences:
    cdef readonly MSA msa

    def __init__(self, MSA msa):
        self.msa = msa

    def __len__(self):
        return self.msa._sequences.size()

    def __getitem__(self, object index):
        cdef ssize_t index_ = index
        cdef ssize_t length = self.msa._sequences.size()

        if index_ < 0:
            index_ += length
        if index_ < 0 or index_ >= length:
            raise IndexError(index)

        return self.msa._sequences[index_].decode()


cdef class _MSANames:
    cdef readonly MSA msa

    def __init__(self, MSA msa):
        self.msa = msa

    def __len__(self):
        return self.msa._names.size()

    def __getitem__(self, object index):
        cdef ssize_t index_ = index
        cdef ssize_t length = self.msa._names.size()

        if index_ < 0:
            index_ += length
        if index_ < 0 or index_ >= length:
            raise IndexError(index)

        return self.msa._names[index_].decode()


cdef class MSA:
    """A multiple sequence alignment.

    Attributes:
        names (`~collections.abc.Sequence` of `str`): The names of the
            sequences in the alignment.
        sequences (`~collections.abc.Sequence` of `str`): The sequences
            in the alignment.

    """
    cdef string         _id
    cdef string         _accession
    cdef vector[string] _names
    cdef vector[string] _sequences
    cdef vector[string] _meta

    cdef readonly _MSASequences sequences
    cdef readonly _MSANames     names

    def __cinit__(self):
        self.names = _MSANames(self)
        self.sequences = _MSASequences(self)

    def __init__(self, object id = "", object accession = "", object names = (), object sequences = ()):
        """__init__(self, id="", accession="", names=(), sequences=())\n--\n

        Create a new MSA object.

        Arguments:
            id (`str`): The identifier of the alignment.
            accession (`str`): The accesion of the alignment.
            names (`~collections.abc.Iterable` of `str`): The names of the
                sequences in the alignment.
            sequences (`~collections.abc.Iterable` of `str` or `bytes`):
                The sequences of the alignment.

        Example:
            >>> msa = pycomsa.MSA(
            ...     id="example_01",
            ...     names=["Sp8", "Sp10", "Sp26", "Sp6", "Sp17", "Sp33"],
            ...     sequences=[
            ...         "-----GLGKVIV-YGIVLGTKSDQFSNWVVWLFPWNGLQIHMMGII",
            ...         "-------DPAVL-FVIMLGTIT-KFS--SEWFFAWLGLEINMMVII",
            ...         "AAAAAAAAALLTYLGLFLGTDYENFA--AAAANAWLGLEINMMAQI",
            ...         "-----ASGAILT-LGIYLFTLCAVIS--VSWYLAWLGLEINMMAII",
            ...         "--FAYTAPDLL-LIGFLLKTVA-TFG--DTWFQLWQGLDLNKMPVF",
            ...         "-------PTILNIAGLHMETDI-NFS--LAWFQAWGGLEINKQAIL",
            ...     ]
            ... )

        Raises:
            `ValueError`: When ``names`` and ``sequences`` do not contain
                the same number of elements, or when ``sequences`` contain
                elements that do not all have the same length.

        Note:
            For better compatibility, all values can be given as Python
            strings (`str`), in which case they will be UTF-8 encoded,
            or any object supporting the buffer-protocol (`bytes`,
            `bytearray`, `memoryview`, `array.array`,
            `pyhmmer.easel.TextSequence`, etc.).

        """
        self._id = to_string(id)
        self._accession = to_string(accession)
        for name, sequence in itertools.zip_longest(names, sequences):
            if name is None or sequence is None:
                raise ValueError("names and sequences must be the same length")
            self._names.push_back(to_string(name))
            self._sequences.push_back(to_string(sequence))

    @property
    def id(self):
        """`str`: The identifier of the MSA.
        """
        return self._id.decode()

    @property
    def accession(self):
        """`str`: The accession of the MSA.
        """
        return self._accession.decode()


# --- FileGuard ----------------------------------------------------------------

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


# --- StockholmReader ----------------------------------------------------------

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

    def close(self):
        with self.guard as file:
            file.detach()

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
        msa._id = self.families[index].ID
        msa._accession = self.families[index].AC

        try:
            with nogil:
                comp.DecompressStockholm(self.data, meta, offsets, msa._names, msa._sequences)
                # Stockholm files compress names with right-justified spaces
                # to make it easier to decompress, but we just want the base
                # name
                for i in range(msa._names.size()):
                    rtrim(msa._names[i])
        except Exception as e:
            raise ValueError("Failed to decompress data") from e

        return msa

    def keys(self):
        return self.index.keys()


class StockholmReader(collections.abc.Sequence):
    """A reader for CoMSA-compressed Stockholm files.
    """

    def __init__(self, object file, str size_format = "N"):
        """Create a new Stockholm reader.

        Arguments:
            file (`~io.IOBase`): A file-like object open for reading
                in binary mode, with support for `~io.IOBase.seek`.
            size_format (`str`): The format to use for reading ``size_t`` 
                values, as a `struct` format specifier. The default 
                ``N`` uses the native ``size_t`` type, but other formats 
                can be given for cross-platform compatibility.

        """
        self._reader = _StockholmReader(file, size_format=size_format)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

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

    def close(self):
        """Close the reader and detach the file-like object.
        """
        self._reader.close()

# --- FastaReader --------------------------------------------------------------

cdef class _FastaReader:

    cdef FileGuard                       guard
    cdef size_t                          length
    cdef vector[uint8_t]                 data

    def __init__(self, object file):
        self.guard = FileGuard(io.BufferedReader(file))
        with self.guard as file:
            self.length = file.seek(0, os.SEEK_END)
        self.data.resize(self.length)

    def close(self):
        with self.guard as file:
            file.detach()

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
        msa._id.clear()
        msa._accession.clear()

        try:
            with nogil:
                comp.DecompressFasta(self.data, msa._names, msa._sequences)
                # FASTA files compress names with a '>' character left of
                # the actual identifier so we can just take a substring
                for i in range(msa._names.size()):
                    if msa._names[i][0] == ord('>'):
                        msa._names[i].erase(msa._names[i].begin())
        except Exception as e:
            raise ValueError("Failed to decompress data") from e

        return msa


class FastaReader(collections.abc.Sequence):
    """A reader for CoMSA-compressed FASTA files.
    """

    def __init__(self, object file):
        """Create a new FASTA reader.

        Arguments:
            file (`~io.IOBase`): A file-like object open for reading
                in binary mode, with support for `~io.IOBase.seek`.

        """
        self._reader = _FastaReader(file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __len__(self):
        return 1

    def __getitem__(self, object index):
        if index > 0 or index < -1:
            raise IndexError(index)
        return self._reader.family()

    def close(self):
        """Close the reader and detach the file-like object.
        """
        self._reader.close()

# --- StockholmWriter ----------------------------------------------------------

cdef class StockholmWriter:
    """A writer for CoMSA-compressed Stockholm files.
    """
    cdef vector[stockholm_family_desc_t] families
    cdef FileGuard                       guard
    cdef str                             size_format
    cdef int                             size_size

    def __init__(self, object file, str size_format = "N"):
        """Create a new Stockholm writer.

        Arguments:
            file (`~io.IOBase`): A file-like object open for writing
                in binary mode.
            size_format (`str`): The format to use for writing ``size_t``
                values, as a `struct` format specifier. The default ``N`` 
                uses the native ``size_t`` type, but other formats can 
                be given for cross-platform compatibility.

        """
        self.guard = FileGuard(io.BufferedWriter(file))
        self.size_format = size_format
        self.size_size = struct.calcsize(size_format)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, MSA msa, bool fast = False):
        """Write a MSA to the compressed file.

        Arguments:
            msa (`~pycomsa.MSA`): The multiple sequence alignment
                to compress and append to the file.
            fast (`bool`): Set to `True` to use *move to front* (MTF)
                instead of *weighted frequency count* (WFC) in the
                compression pipeline. This accelerates compression
                and decompression at the cost of compression efficiency.

        """
        cdef CMSACompress            comp
        cdef vector[uint8_t]         data
        cdef vector[vector[uint8_t]] metadata
        cdef vector[string]          names
        cdef size_t                  comp_text_size
        cdef size_t                  comp_seq_size
        cdef size_t                  offset
        cdef memoryview              mem

        # copy and adjust names
        m = 0
        for name in msa._names:
            m = max(m, name.size())
            names.emplace_back(name)
        for i in range(names.size()):
            names[i].insert(names[i].end(), <size_t> (m + 1 - names[i].size()), <char> ord(' '))

        # add stockholm header
        # NOTE: this is not stricty needed to put in the file, but without
        #       this header the CoMSA decompressor (`CoMSA Sd`) will not 
        #       emit a valid Stockholm file. Out `StockholmReader` however
        #       ignores the metadata block.
        metadata.resize(1)
        for c in b"# STOCKHOLM 1.0\n":
            metadata[0].push_back(c)
        if not msa._id.empty():
            for c in b"# GF ID ":
                metadata[0].push_back(c)
            for i in range(msa._id.size()):
                metadata[0].push_back(msa._id[i])
            metadata[0].push_back(ord(b"\n"))
        if not msa._accession.empty():
            for c in b"# GF AC ":
                metadata[0].push_back(c)
            for i in range(msa._accession.size()):
                metadata[0].push_back(msa._accession[i])
            metadata[0].push_back(ord(b"\n"))

        # compress data
        with nogil:
            comp.CompressStockholm(
                metadata,
                vector[uint32_t](),
                names,
                msa._sequences,
                data,
                comp_text_size,
                comp_seq_size,
                fast,
            )

        # write data to file
        with self.guard as file:
            offset = file.tell()
            # FIXME: avoid using memoryview for PyPy support
            mem = PyMemoryView_FromMemory(<char*> data.data(), data.size(), PyBUF_READ)
            file.write(struct.pack(self.size_format, data.size()))
            file.write(mem)

        # record metadata about current family
        self.families.push_back(stockholm_family_desc_t(
            msa._sequences.size(),
            0 if msa._sequences.empty() else msa._sequences.front().size(),
            0, # TODO: orig_size
            comp_text_size + comp_seq_size,
            offset,
            msa._id,
            msa._accession,
        ))

    def close(self):
        self._write_footer()
        with self.guard as file:
            file.detach()

    cdef size_t _write_uint(self, file, size_t x, bool fixed_size = False):
        cdef uint8_t n_bytes = 0
        cdef size_t  t

        if fixed_size:
            n_bytes = self.size_size
        else:
            while t > 0:
                n_bytes += 1
                t >>= 8
            file.write(struct.pack('B', n_bytes))

        for i in range(n_bytes):
            file.write(struct.pack('B', x & 0xff))
            x >>= 8

        return n_bytes + 0 if fixed_size else 1

    cdef size_t _write_fd(self, file, stockholm_family_desc_t& fd):
        cdef size_t n = 0
        n += self._write_uint(file, fd.n_sequences)
        n += self._write_uint(file, fd.n_columns)
        n += self._write_uint(file, fd.raw_size)
        n += self._write_uint(file, fd.compressed_size)
        n += self._write_uint(file, fd.compressed_data_ptr)
        n += file.write(fd.ID)
        n += file.write(fd.AC)
        return n

    cdef void _write_footer(self):
        with self.guard as file:
            offset = file.tell()
            for fd in self.families:
                self._write_fd(file, fd)
            after = file.tell()
            self._write_uint(file, after - offset, fixed_size = True)


# --- Functions ----------------------------------------------------------------

cdef string to_string(object data):
    cdef string                   output
    cdef const unsigned char[::1] view
    cdef size_t                   i
    if isinstance(data, str):
        data = data.encode()
    view = data
    if view.strides[0] == 1:
        output.resize(view.shape[0])
        if view.shape[0] > 0:
            copy_n( &view[0], view.shape[0], output.begin() )
    else:
        for i in range(view.shape[0]):
            output.push_back(view[i])
    return output


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
def open(file, str mode = "r", str format = None, str size_format = "N"):
    """Open a file for reading or for writing with CoMSA.

    Arguments:
        file (`str`, `~os.PathLike`, or `~io.IOBase`): Either the
            path to a file to be opened, or a file-like object in binary
            mode that supports `~io.IOBase.seek`.
        mode (``r`` or ``w``): The mode with which to open the file,
            similarly to `~builtins.open`.
        format (``fasta``, ``stockholm`` or ``None``): The format of
            the file to be read or written. If `None` given (the default),
            the format will be auto-detected while reading, or set to
            ``stockholm`` when writing.
        size_format (`str`): The format to use for reading and writing
            ``size_t`` values, as a `struct` format specifier. The default
            ``N`` uses the native ``size_t`` type, but other formats can
            be given for cross-platform compatibility.

    Example:
        >>> with pycomsa.open("trimal.msac") as reader:
        ...     msa = reader[0]
        ...     len(msa.sequences)
        50

    Raises:
        `ValueError`: When given an invalid argument, or when the file
            format could not be auto-detected from the file contents.

    """
    if mode == "r":
        if not hasattr(file, "read"):
            file = builtins.open(file, "rb")
            close_file = True
        else:
            file = io.BufferedReader(file)
            close_file = False
        try:
            if format is None:
                format = _detect_format(file, size_format=size_format)
            if format == "fasta":
                reader = FastaReader(file, size_format=size_format)
            elif format == "stockholm":
                reader = StockholmReader(file, size_format=size_format)
            else:
                raise ValueError(f"invalid format: {format!r}")
            yield reader
        finally:
            reader.close()
            if close_file:
                file.close()
    elif mode == "w":
        if not hasattr(file, "write"):
            file = builtins.open(file, "rw")
            close_file = True
        else:
            file = io.BufferedWriter(file)
            close_file = False
        try:
            if format is None:
                format = "stockholm"
            if format == "fasta":
                raise NotImplementedError
            elif format == "stockholm":
                writer = StockholmWriter(file, size_format=size_format)
            else:
                raise ValueError(f"invalid format: {format!r}")
            yield writer
        finally:
            writer.close()
            if close_file:
                file.close()
    else:
        raise ValueError(f"invalid mode: {mode!r}")


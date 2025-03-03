File Formats
============

CoMSA produces files using block compression. The content of the compressed
file varies based on the format of the original file. 

FASTA
-----

FASTA files contains a single compressed block, without additional metadata
such as the compressed block size. FASTA files contain only a single MSA, 
and cannot be concatenated.

.. svgbob::
    :align: center

    +--------------------------+ 
    |                          |  
    |    Compressed block      |  
    |                          |  
    +--------------------------+ 

The sequences are compressed using a combination of pBWT, WFC, RLE and Range 
Coder; the rest of the text (the sequence names) is compressed with LZMA.


Stockholm 
---------

Stockholm files sequences of compressed blocks, preceded by the length of 
each blocks, and terminated by a footer.

.. svgbob::
    :align: center

    +--------------------------+  ^   .------.
    | Compressed block 1 size  |  |   |size_t|
    +--------------------------+  v   '------'
    |                          |  
    |    Compressed block 1    |  
    |                          |  
    +--------------------------+  ^  .------.
    | Compressed block 2 size  |  |  |size_t|
    +--------------------------+  v  '------'
    |                          | 
    |    Compressed block 2    |  
    |                          | 
    +--------------------------+
    |            ...           |
    +--------------------------+  ^  .------.
    | Compressed block n size  |  |  |size_t|
    +--------------------------+  v  '------'
    |                          |  
    |    Compressed block n    |  
    |                          | 
    +--------------------------+
    |          Footer          |
    +--------------------------+  ^  .------.
    |       Footer size        |  |  |size_t|
    +--------------------------+  v  '------'


The footer contains extra metadata for each MSA: the name and accession 
of the full MSA, the compressed and decompressed size, and a pointer in
the file allowing random access.


.. caution::

    Because the footer size are encoded as ``size_t`` integers with native
    byte ordering, the Stockholm files are inherently not portable, and a
    file made on a platform (e.g. x86-64) may not be readable transparently
    on another platform (e.g armv7l). 
    
    `pycomsa` attempts a few checks before loading data, but since CoMSA 
    doesn't emit files with magic bytes, it is not possible to automatically
    detect the format of each file. To mitigate this, the `~pycomsa.open` function
    and the `~pycomsa.StockholmReader` and `~pycomsa.StockholmWriter` types 
    support a ``size_format`` parameter, as a `struct` format character, which 
    allows controlling at runtime in which format the block and footer size
    are loaded.

    For example, the following format always creates files that are 
    using 64-bit little-order endianess for compatibility across 
    platforms:

    .. code::

        >>> with pycomsa.open("output.msac", "w", size_format="<Q") as f:
        ...     f.write(msa)


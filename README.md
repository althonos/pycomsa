# 🗜️ PyCoMSA [![Stars](https://img.shields.io/github/stars/althonos/pycomsa.svg?style=social&maxAge=3600&label=Star)](https://github.com/althonos/pycomsa/stargazers)

*Cython bindings and Python interface to [CoMSA](https://github.com/refresh-bio/CoMSA/), a compressor for multiple-sequence alignments.*

[![Actions](https://img.shields.io/github/actions/workflow/status/althonos/pycomsa/test.yml?branch=main&logo=github&style=flat-square&maxAge=300)](https://github.com/althonos/pycomsa/actions)
[![Coverage](https://img.shields.io/codecov/c/gh/althonos/pycomsa?style=flat-square&maxAge=3600&logo=codecov)](https://codecov.io/gh/althonos/pycomsa/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/gpl-3.0/)
[![PyPI](https://img.shields.io/pypi/v/pycomsa.svg?style=flat-square&maxAge=3600&logo=PyPI)](https://pypi.org/project/pycomsa)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/pycomsa?style=flat-square&maxAge=3600&logo=anaconda)](https://anaconda.org/bioconda/pycomsa)
[![AUR](https://img.shields.io/aur/version/python-pycomsa?logo=archlinux&style=flat-square&maxAge=3600)](https://aur.archlinux.org/packages/python-pycomsa)
[![Wheel](https://img.shields.io/pypi/wheel/pycomsa.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/pycomsa/#files)
[![Python Versions](https://img.shields.io/pypi/pyversions/pycomsa.svg?style=flat-square&maxAge=600&logo=python)](https://pypi.org/project/pycomsa/#files)
[![Python Implementations](https://img.shields.io/pypi/implementation/pycomsa.svg?style=flat-square&maxAge=600&label=impl)](https://pypi.org/project/pycomsa/#files)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/pycomsa/)
[![GitHub issues](https://img.shields.io/github/issues/althonos/pycomsa.svg?style=flat-square&maxAge=600)](https://github.com/althonos/pycomsa/issues)
[![Docs](https://img.shields.io/readthedocs/pycomsa/latest?style=flat-square&maxAge=600)](https://pycomsa.readthedocs.io)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/pycomsa/blob/main/CHANGELOG.md)
[![Downloads](https://img.shields.io/pypi/dm/pycomsa?style=flat-square&color=303f9f&maxAge=86400&label=downloads)](https://pepy.tech/project/pycomsa)


## 🗺️ Overview

CoMSA is a compression method for multiple sequence alignments developed
by [Sebastian Deorowicz](https://github.com/sebastiandeorowicz) *et al.*[\[1\]](#ref1).
It compresses sequence data using a combination of positional Burrows-Wheeler transform (pBWT)[\[2\]](#ref2), weighted-frequency-count transform (WFC)[\[3\]](#ref3),
zero-run-length-encoding transform (RLE), and a range coder[\[4\]](#ref4).
It outperforms general-purpose Lempel-Ziv compression algorithms[\[5\]](#ref5)
for MSA compression.

PyCoMSA is a Python module that provides bindings to CoMSA using
[Cython](https://cython.org/).It directly interacts with the CoMSA
internals, allowing to read files into Python objects, or write
files directly from memory, without having to read or write intermediate,
non-compressed files.

### 📋 Features

The library implements the following features:

- [x] Alignment decoding from FASTA or Stockholm encoded files.
- [x] Automated detection of compressed file formats.
- [x] Indexed access to Stockholm encoded files.
- [x] Compression interface for Stockholm encoded files.


## 🔧 Installing

This project is supported on Python 3.7 and later.

Until a release is made to PyPI, you will have to install the project from
source:

```console
$ pip install pycomsa
```

Check the [*install* page](https://pycomsa.readthedocs.io/en/stable/install.html)
of the documentation for other ways to install PyJess on your machine.

## 💡 Example

Use the [`pycomsa.open`](https://pycomsa.readthedocs.io/en/stable/api/functions.html#pycomsa.open) 
method to open a file for reading, using automatic format detection. The readers implement the
[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)
interface, allowing out-of-order access in the file, or just plain
iteration.

```python
import pycomsa

with pycomsa.open("src/pycomsa/tests/data/trimal.msac") as reader:
    print(len(reader))   # show the number of families in the file
    msa = reader[0]      # load family by positional index
    print(msa.names)     # get the list of files in the alignment
    print(msa.sequences) # get the list of sequences in the alignment
```

To write a MSA to a compressed file, used the same function in writing mode:

```python
with pycomsa.open("test.msac", "w") as writer:
    writer.write(msa)
```

Note that `pycomsa.open` also supports 
[file-like objects](https://docs.python.org/3/glossary.html#term-file-object) 
opened in [binary mode](https://docs.python.org/3/glossary.html#term-binary-file), 
and supporting the `seek` method when reading CoMSA files.

## 💭 Feedback

### ⚠️ Issue Tracker

Found a bug ? Have an enhancement request ? Head over to the [GitHub issue
tracker](https://github.com/althonos/pycomsa/issues) if you need to report
or ask something. If you are filing in on a bug, please include as much
information as you can about the issue, and try to recreate the same bug
in a simple, easily reproducible situation.

### 🏗️ Contributing

Contributions are more than welcome! See
[`CONTRIBUTING.md`](https://github.com/althonos/pycomsa/blob/main/CONTRIBUTING.md)
for more details.

## 📋 Changelog

This project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html)
and provides a [changelog](https://github.com/althonos/pycomsa/blob/main/CHANGELOG.md)
in the [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) format.


## ⚖️ License

This library is provided under the
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).
The CoMSA code was written by [Sebastian Deorowicz](https://github.com/sebastiandeorowicz)
and is distributed under the terms of the GPLv3 as well.

*This project is in no way not affiliated, sponsored, or otherwise endorsed
by the [REFRESH-BIO](https://github.com/refresh-bio) laboratory. It was developed
by [Martin Larralde](https://github.com/althonos/) during his PhD project
at the [Leiden University Medical Center](https://www.lumc.nl/) in
the [Zeller team](https://github.com/zellerlab).*


## 📚 References

- <a id="ref1">\[1\]</a> Deorowicz, S., Walczyszyn, J., & Debudaj-Grabysz, A. (2019). CoMSA: Compression of protein multiple sequence alignment files. Bioinformatics, 35(2), 227–234. [doi:10.1093/bioinformatics/bty619](https://doi.org/10.1093/bioinformatics/bty619)
- <a id="ref2">\[2\]</a> Durbin, R. (2014). Efficient haplotype matching and storage using the positional Burrows–Wheeler transform (PBWT). Bioinformatics, 30(9), 1266–1272. [doi:10.1093/bioinformatics/btu014](https://doi.org/10.1093/bioinformatics/btu014)
- <a id="ref3">\[3\]</a> Deorowicz, S. (2002). Second step algorithms in the Burrows–Wheeler compression algorithm. Software: Practice and Experience, 32(2), 99–111. [doi:10.1002/spe.426](https://doi.org/10.1002/spe.426)
- <a id="ref4">\[4\]</a> Salomon, D., & Motta, G. (2010). Handbook of Data Compression. Springer Science & Business Media. ISBN:978-1-84882-903-9
- <a id="ref5">\[5\]</a> Ziv, J., & Lempel, A. (1977). A universal algorithm for sequential data compression. IEEE Transactions on Information Theory, 23(3), 337–343. IEEE Transactions on Information Theory. [doi:10.1109/TIT.1977.1055714](https://doi.org/10.1109/TIT.1977.1055714)

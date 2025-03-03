# noqa: D104
from . import _comsa
from ._comsa import open, MSA, FastaReader, StockholmReader, StockholmWriter

__author__ = "Martin Larralde <martin.larralde@embl.de>"
__all__ = ["open", "MSA", "FastaReader", "StockholmReader", "StockholmWriter"]
__doc__ = _comsa.__doc__
__version__ = _comsa.__version__

# Small addition to the docstring: we want to show a link redirecting to the
# rendered version of the documentation, but this can only work when Python
# is running with docstrings enabled
if __doc__ is not None:
    __doc__ += """See Also:
    An online rendered version of the documentation for this version
    of the library on
    `Read The Docs <https://pycomsa.readthedocs.io/en/v{}/>`_.

    """.format(
        __version__
    )

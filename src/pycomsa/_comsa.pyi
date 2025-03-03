import typing
from os import PathLike
from typing import BinaryIO, ContextManager, Iterable, Union, Sequence, Optional, Type
from types import TracebackType

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

__version__: str

ByteString = Union[bytes, bytearray, memoryview]
Mode = Literal["r", "w"]

class _MSASequences:
    pass

class _MSANames:
    pass

class MSA:
    def __init__(
        self,
        id: str = "",
        accession: str = "",
        names: Iterable[Union[ByteString, str]] = (),
        sequences: Iterable[Union[ByteString, str]] = (),
    ): ...
    @property
    def id(self) -> str: ...
    @property
    def accession(self) -> str: ...

class StockholmReader(Sequence[MSA]):
    def __init__(self, file: BinaryIO, size_format: str = "N") -> None: ...
    def __enter__(self) -> StockholmReader: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> MSA: ...
    def close(self) -> None: ...

class FastaReader(Sequence[MSA]):
    def __init__(self, file: BinaryIO, size_format: str = "N") -> None: ...
    def __enter__(self) -> FastaReader: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> MSA: ...
    def close(self) -> None: ...

class StockholmWriter:
    def __init__(self, file: BinaryIO, size_format: str = "N") -> None: ...
    def __enter__(self) -> StockholmWriter: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]: ...
    def write(self, msa: MSA, fast: bool = False) -> None: ...
    def close(self) -> None: ...

@typing.overload
def open(
    file: Union[str, PathLike[str], BinaryIO],
    mode: Literal["r"] = "r",
    format: None = None,
    size_format: str = "N",
) -> ContextManager[Union[StockholmReader, FastaReader]]: ...
@typing.overload
def open(
    file: Union[str, PathLike[str], BinaryIO],
    mode: Literal["r"] = "r",
    format: Literal["stockholm"] = None,  # "stockholm",
    size_format: str = "N",
) -> ContextManager[StockholmReader]: ...
@typing.overload
def open(
    file: Union[str, PathLike[str], BinaryIO],
    mode: Literal["r"] = "r",
    format: Literal["fasta"] = None,  # "fasta",
    size_format: str = "N",
) -> ContextManager[FastaReader]: ...
@typing.overload
def open(
    file: Union[str, PathLike[str], BinaryIO],
    mode: Literal["w"] = "w",
    format: Literal["stockholm"] = "stockholm",
    size_format: str = "N",
) -> ContextManager[StockholmWriter]: ...
@typing.overload
def open(
    file: Union[str, PathLike[str], BinaryIO],
    mode: Literal["w"] = "w",
    format: None = None,
    size_format: str = "N",
) -> ContextManager[StockholmWriter]: ...
@typing.overload
def open(
    file: Union[str, PathLike[str], BinaryIO],
    mode: Union[Literal["r"], Literal["w"], None] = "r",
    format: Union[Literal["stockholm"], Literal["fasta"], None] = None,
    size_format: str = "N",
) -> ContextManager[Union[StockholmReader, FastaReader, StockholmWriter]]: ...

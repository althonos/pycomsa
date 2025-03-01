import unittest

from .._comsa import FastaReader, StockholmReader

try:
    from importlib.resources import files as resource_files
except ImportError:
    from importlib_resources import files as resource_files


class TestFastaReader(unittest.TestCase):

    def test_reader(self):
        with resource_files("pycomsa.tests.data").joinpath("example.004.AA.msac").open("rb") as f:
            reader = FastaReader(f)
            self.assertEqual(len(reader), 1)
            msa = reader[0]
            self.assertEqual(len(msa.names), 6)


class TestStockholmReader(unittest.TestCase):

    def test_reader(self):
        lengths = [50,31,33,43,47,35,101,40]
        with resource_files("pycomsa.tests.data").joinpath("trimal.msac").open("rb") as f:
            reader = StockholmReader(f)
            self.assertEqual(len(reader), 8)
            for i, l in enumerate(lengths):
                self.assertEqual(len(reader[i].names), l)

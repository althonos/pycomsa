from . import (
    test_reader,
    test_doctest,
)


def load_tests(loader, suite, pattern):
    suite.addTests(loader.loadTestsFromModule(test_reader))
    test_doctest.load_tests(loader, suite, pattern)
    return suite

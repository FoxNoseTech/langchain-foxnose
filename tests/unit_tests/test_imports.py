"""Validate public API exports."""

import langchain_foxnose
from langchain_foxnose import __all__

EXPECTED_ALL = [
    "FoxNoseBatchWriteError",
    "FoxNoseLoader",
    "FoxNoseRetriever",
    "FoxNoseWriter",
    "__version__",
    "create_foxnose_tool",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)


def test_every_exported_name_is_importable() -> None:
    """__all__ must not advertise a name the package does not actually expose."""
    for name in __all__:
        assert hasattr(langchain_foxnose, name), name

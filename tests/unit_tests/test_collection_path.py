"""Verify folder_path → collection_path migration."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest

from langchain_foxnose import FoxNoseLoader, FoxNoseRetriever, create_foxnose_tool
from langchain_foxnose import _deprecation


def _mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture(autouse=True)
def _reset_warned():
    _deprecation._warned.clear()
    yield
    _deprecation._warned.clear()


# ----- FoxNoseRetriever (Pydantic) -----


def test_retriever_accepts_collection_path() -> None:
    r = FoxNoseRetriever(
        client=_mock_client(),
        collection_path="articles",
        page_content_field="body",
    )
    assert r.collection_path == "articles"


def test_retriever_legacy_folder_path_works_with_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = FoxNoseRetriever(
            client=_mock_client(),
            folder_path="articles",
            page_content_field="body",
        )
    assert r.collection_path == "articles"
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep, "expected DeprecationWarning when using folder_path"
    assert "folder_path" in str(dep[0].message)
    assert "collection_path" in str(dep[0].message)


def test_retriever_rejects_both_kwargs() -> None:
    with pytest.raises(ValueError, match="not both"):
        FoxNoseRetriever(
            client=_mock_client(),
            folder_path="a",
            collection_path="b",
            page_content_field="body",
        )


def test_retriever_folder_path_property_returns_collection_path() -> None:
    r = FoxNoseRetriever(
        client=_mock_client(),
        collection_path="articles",
        page_content_field="body",
    )
    # Backward-compat read access via property.
    assert r.folder_path == "articles"


# ----- FoxNoseLoader (plain class) -----


def test_loader_accepts_collection_path() -> None:
    loader = FoxNoseLoader(
        client=_mock_client(),
        collection_path="kb",
        page_content_field="body",
    )
    assert loader.collection_path == "kb"


def test_loader_legacy_folder_path_works_with_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loader = FoxNoseLoader(
            client=_mock_client(),
            folder_path="kb",
            page_content_field="body",
        )
    assert loader.collection_path == "kb"
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep, "expected DeprecationWarning when using folder_path"
    assert "folder_path" in str(dep[0].message)


def test_loader_rejects_both_kwargs() -> None:
    with pytest.raises(ValueError, match="not both"):
        FoxNoseLoader(
            client=_mock_client(),
            folder_path="a",
            collection_path="b",
            page_content_field="body",
        )


def test_loader_requires_one_of_the_two() -> None:
    with pytest.raises(ValueError, match="collection_path is required"):
        FoxNoseLoader(
            client=_mock_client(),
            page_content_field="body",
        )


def test_loader_folder_path_attribute_backcompat() -> None:
    loader = FoxNoseLoader(
        client=_mock_client(),
        collection_path="kb",
        page_content_field="body",
    )
    # Backward-compat read access via @property.
    assert loader.folder_path == "kb"


# ----- create_foxnose_tool -----


def test_tool_accepts_collection_path() -> None:
    tool = create_foxnose_tool(
        client=_mock_client(),
        collection_path="kb",
        page_content_field="body",
    )
    assert tool is not None


def test_tool_legacy_folder_path_works_with_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tool = create_foxnose_tool(
            client=_mock_client(),
            folder_path="kb",
            page_content_field="body",
        )
    assert tool is not None
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    # The warning is emitted by FoxNoseRetriever when create_foxnose_tool
    # forwards the folder_path kwarg through.
    assert dep, "expected DeprecationWarning to propagate from FoxNoseRetriever"


def test_tool_rejects_both_kwargs() -> None:
    with pytest.raises(ValueError, match="not both"):
        create_foxnose_tool(
            client=_mock_client(),
            folder_path="a",
            collection_path="b",
            page_content_field="body",
        )


# ----- one-shot deprecation semantics -----


def test_deprecation_warning_fires_once_per_process() -> None:
    """Multiple deprecated calls in a row must emit the warning exactly once."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FoxNoseRetriever(
            client=_mock_client(),
            folder_path="kb1",
            page_content_field="body",
        )
        FoxNoseLoader(
            client=_mock_client(),
            folder_path="kb2",
            page_content_field="body",
        )
        create_foxnose_tool(
            client=_mock_client(),
            folder_path="kb3",
            page_content_field="body",
        )
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep) == 1, f"expected one-shot semantics, got {len(dep)} warnings"


# ----- folder_path=None edge case (parity with explicit absence) -----


def test_retriever_treats_folder_path_none_as_absent() -> None:
    """Passing folder_path=None alongside collection_path must not raise.

    Some upstream code may explicitly pass folder_path=None — we should treat
    it as absent rather than as 'deprecated kwarg supplied'.
    """
    r = FoxNoseRetriever(
        client=_mock_client(),
        collection_path="articles",
        folder_path=None,
        page_content_field="body",
    )
    assert r.collection_path == "articles"


def test_loader_treats_folder_path_none_as_absent() -> None:
    loader = FoxNoseLoader(
        client=_mock_client(),
        collection_path="kb",
        folder_path=None,
        page_content_field="body",
    )
    assert loader.collection_path == "kb"


# ----- from_client_params surface (was previously not migrated) -----


def test_retriever_from_client_params_accepts_collection_path(monkeypatch) -> None:
    """The factory must accept the canonical collection_path kwarg."""
    from langchain_foxnose import retrievers as retrievers_mod

    fake_client = _mock_client()

    class _FakeClientCls:
        def __init__(self, **_kwargs):
            pass

    # Stub the SDK module the factory imports at call time.
    fake_module = MagicMock()
    fake_module.FluxClient = _FakeClientCls
    fake_module.AsyncFluxClient = _FakeClientCls
    monkeypatch.setitem(__import__("sys").modules, "foxnose_sdk.flux", fake_module)

    r = FoxNoseRetriever.from_client_params(
        base_url="https://e.fxns.io",
        api_prefix="api",
        auth=fake_client,
        collection_path="articles",
        page_content_field="body",
    )
    assert r.collection_path == "articles"


def test_retriever_from_client_params_accepts_legacy_folder_path(monkeypatch) -> None:
    """The factory must still accept folder_path, with a deprecation warning."""
    from langchain_foxnose import retrievers as retrievers_mod  # noqa: F401

    fake_client = _mock_client()

    class _FakeClientCls:
        def __init__(self, **_kwargs):
            pass

    fake_module = MagicMock()
    fake_module.FluxClient = _FakeClientCls
    fake_module.AsyncFluxClient = _FakeClientCls
    monkeypatch.setitem(__import__("sys").modules, "foxnose_sdk.flux", fake_module)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = FoxNoseRetriever.from_client_params(
            base_url="https://e.fxns.io",
            api_prefix="api",
            auth=fake_client,
            folder_path="articles",
            page_content_field="body",
        )
    assert r.collection_path == "articles"
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep, "expected DeprecationWarning when folder_path is used"


# ----- version drift check -----


def test_version_matches_pyproject() -> None:
    import langchain_foxnose

    assert langchain_foxnose.__version__ == "0.4.0"

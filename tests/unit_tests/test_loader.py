"""Tests for FoxNoseLoader."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_foxnose import FoxNoseLoader
from tests.conftest import SAMPLE_RESULTS, _make_list_response


class TestLoaderValidation:
    """Pydantic model validation."""

    def test_requires_at_least_one_client(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)at least one of"):
            FoxNoseLoader(
                folder_path="kb",
                page_content_field="body",
            )

    def test_requires_content_mapping(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)content mapping strategy"):
            FoxNoseLoader(
                client=MagicMock(),
                folder_path="kb",
            )

    def test_rejects_multiple_content_strategies(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)only one"):
            FoxNoseLoader(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                page_content_fields=["title", "body"],
            )

    def test_rejects_empty_page_content_fields(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)must not be empty"):
            FoxNoseLoader(
                client=MagicMock(),
                folder_path="kb",
                page_content_fields=[],
            )

    def test_rejects_both_metadata_field_options(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)mutually exclusive"):
            FoxNoseLoader(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                metadata_fields=["title"],
                exclude_metadata_fields=["status"],
            )

    def test_rejects_batch_size_zero(self) -> None:
        with pytest.raises((ValueError, Exception), match=r"(?i)batch_size must be"):
            FoxNoseLoader(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                batch_size=0,
            )

    def test_accepts_valid_config(self) -> None:
        loader = FoxNoseLoader(
            client=MagicMock(),
            folder_path="kb",
            page_content_field="body",
        )
        assert loader.folder_path == "kb"
        assert loader.batch_size == 100

    def test_accepts_async_client_only(self) -> None:
        loader = FoxNoseLoader(
            async_client=AsyncMock(),
            folder_path="kb",
            page_content_field="body",
        )
        assert loader.client is None
        assert loader.async_client is not None


class TestLoaderSync:
    """Synchronous loading."""

    def test_load_returns_documents(self, mock_flux_client_with_list: MagicMock) -> None:
        loader = FoxNoseLoader(
            client=mock_flux_client_with_list,
            folder_path="articles",
            page_content_field="body",
        )
        docs = loader.load()
        assert len(docs) == 3
        assert docs[0].page_content == "FoxNose is a serverless knowledge platform..."
        assert docs[0].metadata["key"] == "abc123"

    def test_lazy_load_yields_documents(self, mock_flux_client_with_list: MagicMock) -> None:
        loader = FoxNoseLoader(
            client=mock_flux_client_with_list,
            folder_path="articles",
            page_content_field="body",
        )
        docs = list(loader.lazy_load())
        assert len(docs) == 3

    def test_calls_list_resources(self, mock_flux_client_with_list: MagicMock) -> None:
        loader = FoxNoseLoader(
            client=mock_flux_client_with_list,
            folder_path="articles",
            page_content_field="body",
            batch_size=50,
        )
        loader.load()
        mock_flux_client_with_list.list_resources.assert_called_once()
        call_args = mock_flux_client_with_list.list_resources.call_args
        assert call_args[0][0] == "articles"
        assert call_args[1]["params"]["limit"] == 50

    def test_params_forwarded(self, mock_flux_client_with_list: MagicMock) -> None:
        custom_params: dict[str, Any] = {
            "where": {"status__eq": "published"},
            "sort": "-created_at",
        }
        loader = FoxNoseLoader(
            client=mock_flux_client_with_list,
            folder_path="articles",
            page_content_field="body",
            params=custom_params,
        )
        loader.load()
        call_params = mock_flux_client_with_list.list_resources.call_args[1]["params"]
        assert call_params["where"] == {"status__eq": "published"}
        assert call_params["sort"] == "-created_at"

    def test_pagination_multiple_pages(self) -> None:
        """Cursor-based pagination fetches all pages."""
        page1_results = copy.deepcopy(SAMPLE_RESULTS[:2])
        page2_results = copy.deepcopy(SAMPLE_RESULTS[2:])

        client = MagicMock()
        client.list_resources.side_effect = [
            _make_list_response(page1_results, count=3, next_cursor="cursor_abc"),
            _make_list_response(page2_results, count=3, next_cursor=None),
        ]

        loader = FoxNoseLoader(
            client=client,
            folder_path="articles",
            page_content_field="body",
            batch_size=2,
        )
        docs = loader.load()
        assert len(docs) == 3

        # First call should not have next
        first_call_params = client.list_resources.call_args_list[0][1]["params"]
        assert "next" not in first_call_params
        assert first_call_params["limit"] == 2

        # Second call should include cursor
        second_call_params = client.list_resources.call_args_list[1][1]["params"]
        assert second_call_params["next"] == "cursor_abc"

    def test_empty_results(self) -> None:
        client = MagicMock()
        client.list_resources.return_value = _make_list_response([], count=0)

        loader = FoxNoseLoader(
            client=client,
            folder_path="articles",
            page_content_field="body",
        )
        docs = loader.load()
        assert docs == []

    def test_sync_load_without_sync_client_raises(self) -> None:
        loader = FoxNoseLoader(
            async_client=AsyncMock(),
            folder_path="articles",
            page_content_field="body",
        )
        with pytest.raises(ValueError, match="Synchronous loading requires"):
            loader.load()

    def test_error_propagation(self) -> None:
        client = MagicMock()
        client.list_resources.side_effect = RuntimeError("API error")

        loader = FoxNoseLoader(
            client=client,
            folder_path="articles",
            page_content_field="body",
        )
        with pytest.raises(RuntimeError, match="API error"):
            loader.load()


class TestLoaderContentMapping:
    """Content mapping strategies."""

    def test_single_field(self, mock_flux_client_with_list: MagicMock) -> None:
        loader = FoxNoseLoader(
            client=mock_flux_client_with_list,
            folder_path="articles",
            page_content_field="body",
        )
        docs = loader.load()
        assert docs[0].page_content == "FoxNose is a serverless knowledge platform..."

    def test_multi_field(self, mock_flux_client_with_list: MagicMock) -> None:
        loader = FoxNoseLoader(
            client=mock_flux_client_with_list,
            folder_path="articles",
            page_content_fields=["title", "body"],
            page_content_separator=" | ",
        )
        docs = loader.load()
        assert " | " in docs[0].page_content
        assert "Getting Started with FoxNose" in docs[0].page_content
        assert "FoxNose is a serverless knowledge platform..." in docs[0].page_content

    def test_custom_mapper(self, mock_flux_client_with_list: MagicMock) -> None:
        loader = FoxNoseLoader(
            client=mock_flux_client_with_list,
            folder_path="articles",
            page_content_mapper=lambda r: f"# {r['data']['title']}",
        )
        docs = loader.load()
        assert docs[0].page_content == "# Getting Started with FoxNose"


class TestLoaderMetadata:
    """Metadata field handling."""

    def test_metadata_whitelist(self, mock_flux_client_with_list: MagicMock) -> None:
        loader = FoxNoseLoader(
            client=mock_flux_client_with_list,
            folder_path="articles",
            page_content_field="body",
            metadata_fields=["title"],
        )
        docs = loader.load()
        assert "title" in docs[0].metadata
        assert "category" not in docs[0].metadata

    def test_metadata_blacklist(self, mock_flux_client_with_list: MagicMock) -> None:
        loader = FoxNoseLoader(
            client=mock_flux_client_with_list,
            folder_path="articles",
            page_content_field="body",
            exclude_metadata_fields=["status"],
        )
        docs = loader.load()
        assert "status" not in docs[0].metadata
        assert "title" in docs[0].metadata

    def test_sys_metadata_included_by_default(self, mock_flux_client_with_list: MagicMock) -> None:
        loader = FoxNoseLoader(
            client=mock_flux_client_with_list,
            folder_path="articles",
            page_content_field="body",
        )
        docs = loader.load()
        assert "key" in docs[0].metadata
        assert "folder" in docs[0].metadata

    def test_sys_metadata_excluded(self, mock_flux_client_with_list: MagicMock) -> None:
        loader = FoxNoseLoader(
            client=mock_flux_client_with_list,
            folder_path="articles",
            page_content_field="body",
            include_sys_metadata=False,
        )
        docs = loader.load()
        assert "key" not in docs[0].metadata
        assert "folder" not in docs[0].metadata


class TestLoaderAsync:
    """Asynchronous loading."""

    async def test_alazy_load_returns_documents(
        self, mock_async_flux_client_with_list: AsyncMock
    ) -> None:
        loader = FoxNoseLoader(
            async_client=mock_async_flux_client_with_list,
            folder_path="articles",
            page_content_field="body",
        )
        docs = [doc async for doc in loader.alazy_load()]
        assert len(docs) == 3
        assert docs[0].page_content == "FoxNose is a serverless knowledge platform..."

    async def test_alazy_load_pagination(self) -> None:
        """Async cursor-based pagination fetches all pages."""
        page1_results = copy.deepcopy(SAMPLE_RESULTS[:2])
        page2_results = copy.deepcopy(SAMPLE_RESULTS[2:])

        client = AsyncMock()
        client.list_resources.side_effect = [
            _make_list_response(page1_results, count=3, next_cursor="cursor_xyz"),
            _make_list_response(page2_results, count=3, next_cursor=None),
        ]

        loader = FoxNoseLoader(
            async_client=client,
            folder_path="articles",
            page_content_field="body",
            batch_size=2,
        )
        docs = [doc async for doc in loader.alazy_load()]
        assert len(docs) == 3

        # Verify cursor was passed on second call
        second_call_params = client.list_resources.call_args_list[1][1]["params"]
        assert second_call_params["next"] == "cursor_xyz"

    async def test_alazy_load_without_async_client_raises(self) -> None:
        loader = FoxNoseLoader(
            client=MagicMock(),
            folder_path="articles",
            page_content_field="body",
        )
        with pytest.raises(ValueError, match="Async loading requires"):
            async for _ in loader.alazy_load():
                pass

    async def test_alazy_load_empty_results(self) -> None:
        client = AsyncMock()
        client.list_resources.return_value = _make_list_response([], count=0)

        loader = FoxNoseLoader(
            async_client=client,
            folder_path="articles",
            page_content_field="body",
        )
        docs = [doc async for doc in loader.alazy_load()]
        assert docs == []

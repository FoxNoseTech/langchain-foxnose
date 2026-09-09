"""Query-string parameter passthrough (query_params / truncate_text)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_foxnose import FoxNoseRetriever


def _retriever(client: Any, **kwargs: Any) -> FoxNoseRetriever:
    return FoxNoseRetriever(
        client=client,
        collection_path="articles",
        page_content_field="body",
        **kwargs,
    )


class TestTruncateTextValidation:
    def test_rejects_zero(self, mock_flux_client: MagicMock) -> None:
        with pytest.raises(ValueError, match="truncate_text must be >= 1"):
            _retriever(mock_flux_client, truncate_text=0)

    def test_rejects_negative(self, mock_flux_client: MagicMock) -> None:
        with pytest.raises(ValueError, match="truncate_text must be >= 1"):
            _retriever(mock_flux_client, truncate_text=-5)

    def test_rejects_duplicate_in_query_params(self, mock_flux_client: MagicMock) -> None:
        with pytest.raises(ValueError, match="truncate_text is set both"):
            _retriever(
                mock_flux_client,
                truncate_text=100,
                query_params={"truncate_text": 200},
            )

    def test_rejects_truncate_text_in_search_kwargs(self, mock_flux_client: MagicMock) -> None:
        with pytest.raises(ValueError, match="query-string parameters"):
            _retriever(mock_flux_client, search_kwargs={"truncate_text": 200})

    def test_rejects_query_params_in_search_kwargs(self, mock_flux_client: MagicMock) -> None:
        with pytest.raises(ValueError, match="query-string parameters"):
            _retriever(mock_flux_client, search_kwargs={"query_params": {"a": 1}})


class TestTruncateTextPassthrough:
    def test_text_mode_passes_params(self, mock_flux_client: MagicMock) -> None:
        _retriever(mock_flux_client, search_mode="text", truncate_text=120).invoke("q")
        assert mock_flux_client.search.call_args.kwargs["params"] == {"truncate_text": 120}

    def test_vector_mode_passes_query_params(self, mock_flux_client: MagicMock) -> None:
        _retriever(mock_flux_client, search_mode="vector", truncate_text=120).invoke("q")
        kwargs = mock_flux_client.vector_search.call_args.kwargs
        assert kwargs["query_params"] == {"truncate_text": 120}

    def test_hybrid_mode_passes_query_params(self, mock_flux_client: MagicMock) -> None:
        _retriever(mock_flux_client, search_mode="hybrid", truncate_text=120).invoke("q")
        kwargs = mock_flux_client.hybrid_search.call_args.kwargs
        assert kwargs["query_params"] == {"truncate_text": 120}

    def test_boosted_mode_passes_query_params(self, mock_flux_client: MagicMock) -> None:
        _retriever(mock_flux_client, search_mode="vector_boosted", truncate_text=120).invoke("q")
        kwargs = mock_flux_client.boosted_search.call_args.kwargs
        assert kwargs["query_params"] == {"truncate_text": 120}

    def test_vector_field_mode_passes_query_params(self, mock_flux_client: MagicMock) -> None:
        _retriever(
            mock_flux_client,
            search_mode="vector",
            vector_field="embedding",
            query_vector=[0.1, 0.2],
            truncate_text=120,
        ).invoke("q")
        kwargs = mock_flux_client.vector_field_search.call_args.kwargs
        assert kwargs["query_params"] == {"truncate_text": 120}

    def test_query_params_merged_with_truncate_text(self, mock_flux_client: MagicMock) -> None:
        _retriever(
            mock_flux_client,
            search_mode="text",
            truncate_text=50,
            query_params={"locale": "en"},
        ).invoke("q")
        assert mock_flux_client.search.call_args.kwargs["params"] == {
            "locale": "en",
            "truncate_text": 50,
        }

    def test_none_when_unset(self, mock_flux_client: MagicMock) -> None:
        _retriever(mock_flux_client, search_mode="text").invoke("q")
        assert mock_flux_client.search.call_args.kwargs["params"] is None

    def test_query_params_not_mutated(self, mock_flux_client: MagicMock) -> None:
        qp: dict[str, Any] = {"locale": "en"}
        _retriever(mock_flux_client, search_mode="text", truncate_text=50, query_params=qp).invoke(
            "q"
        )
        assert qp == {"locale": "en"}


class TestAsyncTruncateTextPassthrough:
    """The async call sites are duplicated code — cover every one of them."""

    async def _invoke(self, client: Any, **kwargs: Any) -> None:
        retriever = FoxNoseRetriever(
            async_client=client,
            collection_path="articles",
            page_content_field="body",
            truncate_text=77,
            **kwargs,
        )
        await retriever.ainvoke("q")

    async def test_text_mode(self, mock_async_flux_client: Any) -> None:
        await self._invoke(mock_async_flux_client, search_mode="text")
        assert mock_async_flux_client.search.call_args.kwargs["params"] == {"truncate_text": 77}

    async def test_vector_mode(self, mock_async_flux_client: Any) -> None:
        await self._invoke(mock_async_flux_client, search_mode="vector")
        assert mock_async_flux_client.vector_search.call_args.kwargs["query_params"] == {
            "truncate_text": 77
        }

    async def test_vector_field_mode(self, mock_async_flux_client: Any) -> None:
        await self._invoke(
            mock_async_flux_client,
            search_mode="vector",
            vector_field="embedding",
            query_vector=[0.1, 0.2],
        )
        kwargs = mock_async_flux_client.vector_field_search.call_args.kwargs
        assert kwargs["query_params"] == {"truncate_text": 77}

    async def test_hybrid_mode(self, mock_async_flux_client: Any) -> None:
        await self._invoke(mock_async_flux_client, search_mode="hybrid")
        assert mock_async_flux_client.hybrid_search.call_args.kwargs["query_params"] == {
            "truncate_text": 77
        }

    async def test_boosted_mode(self, mock_async_flux_client: Any) -> None:
        await self._invoke(mock_async_flux_client, search_mode="vector_boosted")
        assert mock_async_flux_client.boosted_search.call_args.kwargs["query_params"] == {
            "truncate_text": 77
        }

    async def test_none_when_unset(self, mock_async_flux_client: Any) -> None:
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            collection_path="articles",
            page_content_field="body",
            search_mode="text",
        )
        await retriever.ainvoke("q")
        assert mock_async_flux_client.search.call_args.kwargs["params"] is None

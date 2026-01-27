"""Tests for FoxNoseRetriever (async)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_foxnose import FoxNoseRetriever


class TestRetrieverAsync:
    """Async retrieval with AsyncFluxClient."""

    @pytest.mark.asyncio
    async def test_ainvoke_with_async_client(self, mock_async_flux_client: AsyncMock) -> None:
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="hybrid",
            top_k=5,
        )
        docs = await retriever.ainvoke("test query")
        assert len(docs) == 3
        assert docs[0].page_content == "FoxNose is a serverless knowledge platform..."
        assert docs[0].metadata["key"] == "abc123"

    @pytest.mark.asyncio
    async def test_ainvoke_calls_async_search(self, mock_async_flux_client: AsyncMock) -> None:
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector",
            top_k=20,
        )
        await retriever.ainvoke("semantic query")
        mock_async_flux_client.search.assert_called_once()
        call_args = mock_async_flux_client.search.call_args
        assert call_args[0][0] == "articles"
        body = call_args[1]["body"]
        assert body["search_mode"] == "vector"
        assert "vector_search" in body
        assert "find_text" not in body

    @pytest.mark.asyncio
    async def test_ainvoke_with_both_clients_uses_async(
        self,
        mock_flux_client: MagicMock,
        mock_async_flux_client: AsyncMock,
    ) -> None:
        """When both clients are provided, async path uses async_client."""
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
        )
        docs = await retriever.ainvoke("query")
        # Async client should be called, not sync
        mock_async_flux_client.search.assert_called_once()
        mock_flux_client.search.assert_not_called()
        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_ainvoke_empty_results(self, mock_async_flux_client: AsyncMock) -> None:
        mock_async_flux_client.search.return_value = {"results": [], "limit": 5}
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
        )
        docs = await retriever.ainvoke("query")
        assert docs == []

    @pytest.mark.asyncio
    async def test_ainvoke_error_propagation(self, mock_async_flux_client: AsyncMock) -> None:
        mock_async_flux_client.search.side_effect = RuntimeError("Async API error")
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
        )
        with pytest.raises(RuntimeError, match="Async API error"):
            await retriever.ainvoke("query")

    @pytest.mark.asyncio
    async def test_ainvoke_with_where_filter(self, mock_async_flux_client: AsyncMock) -> None:
        where_filter = {"$": {"all_of": [{"status__eq": "published"}]}}
        retriever = FoxNoseRetriever(
            async_client=mock_async_flux_client,
            folder_path="articles",
            page_content_field="body",
            where=where_filter,
        )
        await retriever.ainvoke("query")
        body = mock_async_flux_client.search.call_args[1]["body"]
        assert body["where"] == where_filter

    @pytest.mark.asyncio
    async def test_ainvoke_fallback_to_sync(
        self,
        mock_flux_client: MagicMock,
    ) -> None:
        """When only sync client is provided, ainvoke should fall back to sync via executor."""
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
        )
        docs = await retriever.ainvoke("query")
        # Should have called sync client via executor fallback
        mock_flux_client.search.assert_called_once()
        assert len(docs) == 3

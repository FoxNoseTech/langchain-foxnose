"""Tests for FoxNoseRetriever (synchronous)."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from langchain_foxnose import FoxNoseRetriever


class _MockEmbeddings(Embeddings):
    """Minimal Embeddings mock that satisfies Pydantic type checking."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self._vector = vector or [0.1, 0.2, 0.3]
        self.embed_query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        return self._vector


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestRetrieverValidation:
    """Pydantic model validation."""

    def test_requires_at_least_one_client(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)at least one of"):
            FoxNoseRetriever(
                folder_path="kb",
                page_content_field="body",
            )

    def test_requires_content_mapping(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)content mapping strategy"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
            )

    def test_rejects_multiple_content_strategies(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)only one"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                page_content_fields=["title", "body"],
            )

    def test_rejects_invalid_search_mode(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)invalid search_mode"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="invalid",
            )

    def test_rejects_top_k_zero(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)top_k must be"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                top_k=0,
            )

    def test_rejects_negative_top_k(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)top_k must be"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                top_k=-5,
            )

    def test_constructor_k_alias(self) -> None:
        """LangChain standard tests construct with k=N."""
        retriever = FoxNoseRetriever(
            client=MagicMock(),
            folder_path="kb",
            page_content_field="body",
            k=3,
        )
        assert retriever.top_k == 3

    def test_constructor_rejects_both_k_and_top_k(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)cannot pass both"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                k=3,
                top_k=5,
            )

    def test_rejects_both_metadata_field_options(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)mutually exclusive"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                metadata_fields=["title"],
                exclude_metadata_fields=["status"],
            )

    def test_rejects_empty_page_content_fields(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)must not be empty"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_fields=[],
            )

    def test_rejects_text_threshold_out_of_range(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)text_threshold must be"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                text_threshold=1.5,
            )

    def test_rejects_negative_text_threshold(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)text_threshold must be"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                text_threshold=-0.1,
            )

    def test_rejects_similarity_threshold_out_of_range(self) -> None:
        with pytest.raises(
            (ValueError, ValidationError), match=r"(?i)similarity_threshold must be"
        ):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                similarity_threshold=2.0,
            )

    def test_rejects_conflicting_search_kwargs(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)must not contain"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_kwargs={"search_mode": "text"},
            )

    def test_rejects_vector_search_in_search_kwargs(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)must not contain"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_kwargs={"vector_search": {"query": "x"}},
            )

    def test_accepts_valid_config(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="kb",
            page_content_field="body",
            search_mode="hybrid",
        )
        assert retriever.folder_path == "kb"
        assert retriever.search_mode == "hybrid"

    def test_accepts_async_client_only(self) -> None:
        from unittest.mock import AsyncMock

        retriever = FoxNoseRetriever(
            async_client=AsyncMock(),
            folder_path="kb",
            page_content_field="body",
        )
        assert retriever.client is None
        assert retriever.async_client is not None

    def test_accepts_page_content_mapper(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="kb",
            page_content_mapper=lambda r: r["data"]["body"],
        )
        assert retriever.page_content_mapper is not None


# ---------------------------------------------------------------------------
# Custom embeddings validation
# ---------------------------------------------------------------------------


class TestRetrieverEmbeddingsValidation:
    """Validation for embeddings / query_vector / vector_field."""

    def test_rejects_embeddings_without_vector_field(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)vector_field.*required"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="vector",
                embeddings=_MockEmbeddings(),
            )

    def test_rejects_query_vector_without_vector_field(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)vector_field.*required"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="vector",
                query_vector=[0.1, 0.2, 0.3],
            )

    def test_rejects_embeddings_and_query_vector_together(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)mutually exclusive"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="vector",
                embeddings=_MockEmbeddings(),
                query_vector=[0.1, 0.2],
                vector_field="embedding",
            )

    def test_rejects_vector_field_without_source(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)requires.*embeddings"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="vector",
                vector_field="embedding",
            )

    def test_rejects_embeddings_in_text_mode(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)only supported in"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="text",
                embeddings=_MockEmbeddings(),
                vector_field="embedding",
            )

    def test_rejects_embeddings_in_hybrid_mode(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)only supported in"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="hybrid",
                embeddings=_MockEmbeddings(),
                vector_field="embedding",
            )

    def test_rejects_vector_field_and_vector_fields(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)mutually exclusive"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="vector",
                embeddings=_MockEmbeddings(),
                vector_field="embedding",
                vector_fields=["embedding"],
            )

    def test_rejects_empty_query_vector(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)must not be empty"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="vector",
                query_vector=[],
                vector_field="embedding",
            )

    def test_rejects_nan_in_query_vector(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)finite"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="vector",
                query_vector=[0.1, float("nan"), 0.3],
                vector_field="embedding",
            )

    def test_rejects_inf_in_query_vector(self) -> None:
        with pytest.raises((ValueError, ValidationError), match=r"(?i)finite"):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="vector",
                query_vector=[0.1, math.inf, 0.3],
                vector_field="embedding",
            )

    def test_accepts_embeddings_in_vector_mode(self) -> None:
        retriever = FoxNoseRetriever(
            client=MagicMock(),
            folder_path="kb",
            page_content_field="body",
            search_mode="vector",
            embeddings=_MockEmbeddings(),
            vector_field="embedding",
        )
        assert retriever.embeddings is not None
        assert retriever.vector_field == "embedding"

    def test_accepts_query_vector_in_vector_boosted_mode(self) -> None:
        retriever = FoxNoseRetriever(
            client=MagicMock(),
            folder_path="kb",
            page_content_field="body",
            search_mode="vector_boosted",
            query_vector=[0.1, 0.2, 0.3],
            vector_field="embedding",
        )
        assert retriever.query_vector == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Config dict validation (strict models)
# ---------------------------------------------------------------------------


class TestRetrieverConfigValidation:
    """Strict validation of hybrid_config / vector_boost_config dicts."""

    def test_rejects_unknown_key_in_hybrid_config(self) -> None:
        with pytest.raises((ValueError, ValidationError)):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="hybrid",
                hybrid_config={"vector_weight": 0.6, "text_weight": 0.4, "typo_key": True},
            )

    def test_rejects_invalid_weights_in_hybrid_config(self) -> None:
        with pytest.raises((ValueError, ValidationError)):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="hybrid",
                hybrid_config={"vector_weight": 0.8, "text_weight": 0.8},
            )

    def test_rejects_unknown_key_in_vector_boost_config(self) -> None:
        with pytest.raises((ValueError, ValidationError)):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="vector_boosted",
                vector_boost_config={"boost_factor": 1.5, "unknown_key": 42},
            )

    def test_rejects_invalid_boost_factor(self) -> None:
        with pytest.raises((ValueError, ValidationError)):
            FoxNoseRetriever(
                client=MagicMock(),
                folder_path="kb",
                page_content_field="body",
                search_mode="vector_boosted",
                vector_boost_config={"boost_factor": -1.0},
            )

    def test_accepts_valid_hybrid_config(self) -> None:
        retriever = FoxNoseRetriever(
            client=MagicMock(),
            folder_path="kb",
            page_content_field="body",
            search_mode="hybrid",
            hybrid_config={"vector_weight": 0.7, "text_weight": 0.3},
        )
        assert retriever.hybrid_config == {"vector_weight": 0.7, "text_weight": 0.3}

    def test_accepts_valid_vector_boost_config(self) -> None:
        retriever = FoxNoseRetriever(
            client=MagicMock(),
            folder_path="kb",
            page_content_field="body",
            search_mode="vector_boosted",
            vector_boost_config={"boost_factor": 2.0, "max_boost_results": 10},
        )
        assert retriever.vector_boost_config["boost_factor"] == 2.0


# ---------------------------------------------------------------------------
# Sync retrieval — SDK method dispatch
# ---------------------------------------------------------------------------


class TestRetrieverSync:
    """Synchronous retrieval and SDK method dispatch."""

    def test_invoke_returns_documents(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="hybrid",
            top_k=5,
        )
        docs = retriever.invoke("test query")
        assert len(docs) == 3
        assert docs[0].page_content == "FoxNose is a serverless knowledge platform..."
        assert docs[0].metadata["key"] == "abc123"

    def test_text_mode_calls_client_search(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="text",
            top_k=10,
        )
        retriever.invoke("keyword search")
        mock_flux_client.search.assert_called_once()
        body = mock_flux_client.search.call_args[1]["body"]
        assert body["search_mode"] == "text"
        assert body["find_text"]["query"] == "keyword search"
        assert body["limit"] == 10
        assert "vector_search" not in body

    def test_text_mode_includes_search_fields_and_threshold(
        self, mock_flux_client: MagicMock
    ) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="text",
            search_fields=["title", "body"],
            text_threshold=0.85,
        )
        retriever.invoke("query")
        body = mock_flux_client.search.call_args[1]["body"]
        assert body["find_text"]["fields"] == ["title", "body"]
        assert body["find_text"]["threshold"] == 0.85

    def test_vector_mode_calls_vector_search(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector",
            top_k=20,
            similarity_threshold=0.8,
        )
        retriever.invoke("semantic search")
        mock_flux_client.vector_search.assert_called_once()
        kwargs = mock_flux_client.vector_search.call_args[1]
        assert kwargs["query"] == "semantic search"
        assert kwargs["top_k"] == 20
        assert kwargs["similarity_threshold"] == 0.8
        mock_flux_client.search.assert_not_called()

    def test_vector_mode_passes_vector_fields(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector",
            vector_fields=["description"],
        )
        retriever.invoke("query")
        kwargs = mock_flux_client.vector_search.call_args[1]
        assert kwargs["fields"] == ["description"]

    def test_hybrid_mode_calls_hybrid_search(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="hybrid",
            top_k=5,
            hybrid_config={"vector_weight": 0.7, "text_weight": 0.3, "rerank_results": False},
        )
        retriever.invoke("test query")
        mock_flux_client.hybrid_search.assert_called_once()
        kwargs = mock_flux_client.hybrid_search.call_args[1]
        assert kwargs["query"] == "test query"
        assert kwargs["find_text"]["query"] == "test query"
        assert kwargs["vector_weight"] == 0.7
        assert kwargs["text_weight"] == 0.3
        assert kwargs["rerank_results"] is False

    def test_boosted_mode_calls_boosted_search(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector_boosted",
            top_k=10,
            vector_boost_config={"boost_factor": 2.0, "max_boost_results": 15},
        )
        retriever.invoke("boosted query")
        mock_flux_client.boosted_search.assert_called_once()
        kwargs = mock_flux_client.boosted_search.call_args[1]
        assert kwargs["query"] == "boosted query"
        assert kwargs["find_text"]["query"] == "boosted query"
        assert kwargs["boost_factor"] == 2.0
        assert kwargs["max_boost_results"] == 15

    def test_where_filter_passed(self, mock_flux_client: MagicMock) -> None:
        where_filter = {"$": {"all_of": [{"status__eq": "published"}]}}
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            where=where_filter,
        )
        retriever.invoke("query")
        kwargs = mock_flux_client.hybrid_search.call_args[1]
        assert kwargs["where"] == where_filter

    def test_sort_passed(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            sort=["-_sys.created_at"],
        )
        retriever.invoke("query")
        kwargs = mock_flux_client.hybrid_search.call_args[1]
        assert kwargs["sort"] == ["-_sys.created_at"]

    def test_search_kwargs_limit_as_named_param(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_kwargs={"limit": 50},
        )
        retriever.invoke("query")
        kwargs = mock_flux_client.hybrid_search.call_args[1]
        assert kwargs["limit"] == 50

    def test_search_kwargs_extra_body(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_kwargs={"ignore_unknown_fields": True},
        )
        retriever.invoke("query")
        kwargs = mock_flux_client.hybrid_search.call_args[1]
        assert kwargs["ignore_unknown_fields"] is True

    def test_search_kwargs_text_mode(self, mock_flux_client: MagicMock) -> None:
        """In text mode, search_kwargs limit overrides top_k in body."""
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="text",
            top_k=5,
            search_kwargs={"limit": 50, "ignore_unknown_fields": True},
        )
        retriever.invoke("query")
        body = mock_flux_client.search.call_args[1]["body"]
        assert body["limit"] == 50
        assert body["ignore_unknown_fields"] is True

    def test_text_mode_with_offset_and_sort(self, mock_flux_client: MagicMock) -> None:
        """Text mode passes offset and sort through to the body."""
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="text",
            sort=["-_sys.created_at"],
            search_kwargs={"offset": 10},
        )
        retriever.invoke("query")
        body = mock_flux_client.search.call_args[1]["body"]
        assert body["offset"] == 10
        assert body["sort"] == ["-_sys.created_at"]

    def test_search_kwargs_where_override_text_mode(self, mock_flux_client: MagicMock) -> None:
        """search_kwargs where/sort override instance-level values in text mode."""
        override_where = {"$": {"all_of": [{"category__eq": "override"}]}}
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="text",
            where={"$": {"all_of": [{"status__eq": "published"}]}},
            search_kwargs={"where": override_where, "sort": ["-title"]},
        )
        retriever.invoke("query")
        body = mock_flux_client.search.call_args[1]["body"]
        assert body["where"] == override_where
        assert body["sort"] == ["-title"]

    def test_multi_field_content(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_fields=["title", "body"],
            page_content_separator=" | ",
        )
        docs = retriever.invoke("query")
        assert " | " in docs[0].page_content
        assert "Getting Started with FoxNose" in docs[0].page_content

    def test_custom_mapper(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_mapper=lambda r: f"# {r['data']['title']}",
        )
        docs = retriever.invoke("query")
        assert docs[0].page_content == "# Getting Started with FoxNose"

    def test_metadata_whitelist(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            metadata_fields=["title"],
        )
        docs = retriever.invoke("query")
        assert "title" in docs[0].metadata
        assert "category" not in docs[0].metadata

    def test_metadata_blacklist(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            exclude_metadata_fields=["status"],
        )
        docs = retriever.invoke("query")
        assert "status" not in docs[0].metadata
        assert "title" in docs[0].metadata

    def test_empty_results(self, mock_flux_client: MagicMock) -> None:
        mock_flux_client.hybrid_search.return_value = {"results": [], "limit": 5}
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
        )
        docs = retriever.invoke("query")
        assert docs == []

    def test_error_propagation(self, mock_flux_client: MagicMock) -> None:
        mock_flux_client.hybrid_search.side_effect = RuntimeError("API error")
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
        )
        with pytest.raises(RuntimeError, match="API error"):
            retriever.invoke("query")

    def test_sync_invoke_without_sync_client_raises(self) -> None:
        from unittest.mock import AsyncMock

        retriever = FoxNoseRetriever(
            async_client=AsyncMock(),
            folder_path="articles",
            page_content_field="body",
        )
        with pytest.raises(ValueError, match="Synchronous retrieval requires"):
            retriever.invoke("query")


# ---------------------------------------------------------------------------
# Vector field search (custom embeddings)
# ---------------------------------------------------------------------------


class TestRetrieverVectorFieldSearch:
    """Custom embedding search via vector_field_search."""

    def test_vector_mode_with_embeddings(self, mock_flux_client: MagicMock) -> None:
        embeddings = _MockEmbeddings(vector=[0.1, 0.2, 0.3])

        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector",
            embeddings=embeddings,
            vector_field="embedding",
            top_k=10,
            similarity_threshold=0.75,
        )
        docs = retriever.invoke("test query")

        assert embeddings.embed_query_calls == ["test query"]
        mock_flux_client.vector_field_search.assert_called_once()
        kwargs = mock_flux_client.vector_field_search.call_args[1]
        assert kwargs["field"] == "embedding"
        assert kwargs["query_vector"] == [0.1, 0.2, 0.3]
        assert kwargs["top_k"] == 10
        assert kwargs["similarity_threshold"] == 0.75
        assert len(docs) == 3

    def test_vector_mode_with_static_query_vector(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector",
            query_vector=[0.5, 0.6, 0.7],
            vector_field="embedding",
        )
        retriever.invoke("ignored query text")

        mock_flux_client.vector_field_search.assert_called_once()
        kwargs = mock_flux_client.vector_field_search.call_args[1]
        assert kwargs["query_vector"] == [0.5, 0.6, 0.7]
        assert kwargs["field"] == "embedding"

    def test_boosted_mode_with_embeddings(self, mock_flux_client: MagicMock) -> None:
        embeddings = _MockEmbeddings(vector=[0.1, 0.2])

        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector_boosted",
            embeddings=embeddings,
            vector_field="embedding",
        )
        retriever.invoke("boosted query")

        assert embeddings.embed_query_calls == ["boosted query"]
        mock_flux_client.boosted_search.assert_called_once()
        kwargs = mock_flux_client.boosted_search.call_args[1]
        assert kwargs["field"] == "embedding"
        assert kwargs["query_vector"] == [0.1, 0.2]
        assert "query" not in kwargs  # Should not send text query for custom embeddings

    def test_boosted_mode_with_static_vector(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector_boosted",
            query_vector=[0.9, 0.8],
            vector_field="embedding",
        )
        retriever.invoke("query")

        kwargs = mock_flux_client.boosted_search.call_args[1]
        assert kwargs["field"] == "embedding"
        assert kwargs["query_vector"] == [0.9, 0.8]


# ---------------------------------------------------------------------------
# Runtime top_k override
# ---------------------------------------------------------------------------


class TestRetrieverRuntimeTopK:
    """Runtime top_k override via invoke(..., top_k=N)."""

    def test_invoke_runtime_top_k_hybrid(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="hybrid",
            top_k=10,
        )
        retriever.invoke("query", top_k=3)
        kwargs = mock_flux_client.hybrid_search.call_args[1]
        assert kwargs["top_k"] == 3

    def test_invoke_runtime_top_k_text(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="text",
            top_k=10,
        )
        retriever.invoke("query", top_k=1)
        body = mock_flux_client.search.call_args[1]["body"]
        assert body["limit"] == 1

    def test_invoke_runtime_top_k_vector(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="vector",
            top_k=10,
        )
        retriever.invoke("query", top_k=2)
        kwargs = mock_flux_client.vector_search.call_args[1]
        assert kwargs["top_k"] == 2

    def test_invoke_without_runtime_top_k_uses_default(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="hybrid",
            top_k=7,
        )
        retriever.invoke("query")
        kwargs = mock_flux_client.hybrid_search.call_args[1]
        assert kwargs["top_k"] == 7

    def test_invoke_runtime_k_alias(self, mock_flux_client: MagicMock) -> None:
        """LangChain standard tests use 'k' as the default arg name."""
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="hybrid",
            top_k=10,
        )
        retriever.invoke("query", k=2)
        kwargs = mock_flux_client.hybrid_search.call_args[1]
        assert kwargs["top_k"] == 2

    def test_invoke_runtime_top_k_validation(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="hybrid",
        )
        with pytest.raises(ValueError, match="must be an integer >= 1"):
            retriever.invoke("query", top_k=0)
        with pytest.raises(ValueError, match="must be an integer >= 1"):
            retriever.invoke("query", top_k=-1)

"""Tests for the search body builder."""

from __future__ import annotations

import pytest

from langchain_foxnose._search import build_search_body


class TestBuildSearchBodyDeprecation:
    """Deprecation warning."""

    def test_emits_deprecation_warning(self) -> None:
        with pytest.warns(DeprecationWarning, match=r"(?i)deprecated"):
            build_search_body("hello", search_mode="text", top_k=5)


class TestBuildSearchBodyTextMode:
    """Text search mode."""

    def test_text_mode_basic(self) -> None:
        body = build_search_body("hello", search_mode="text", top_k=10)
        assert body == {
            "search_mode": "text",
            "limit": 10,
            "find_text": {"query": "hello"},
        }

    def test_text_mode_with_fields_and_threshold(self) -> None:
        body = build_search_body(
            "machine learning",
            search_mode="text",
            top_k=5,
            search_fields=["title", "summary"],
            text_threshold=0.85,
        )
        assert body["find_text"] == {
            "query": "machine learning",
            "fields": ["title", "summary"],
            "threshold": 0.85,
        }

    def test_text_mode_has_no_vector_search(self) -> None:
        body = build_search_body("query", search_mode="text")
        assert "vector_search" not in body


class TestBuildSearchBodyVectorMode:
    """Vector search mode."""

    def test_vector_mode_basic(self) -> None:
        body = build_search_body("semantic query", search_mode="vector", top_k=20)
        assert body == {
            "search_mode": "vector",
            "limit": 20,
            "vector_search": {"query": "semantic query", "top_k": 20},
        }

    def test_vector_mode_has_no_find_text(self) -> None:
        body = build_search_body("query", search_mode="vector")
        assert "find_text" not in body

    def test_vector_mode_with_fields_and_threshold(self) -> None:
        body = build_search_body(
            "cozy room",
            search_mode="vector",
            top_k=30,
            vector_fields=["description"],
            similarity_threshold=0.65,
        )
        assert body["vector_search"] == {
            "query": "cozy room",
            "top_k": 30,
            "fields": ["description"],
            "similarity_threshold": 0.65,
        }


class TestBuildSearchBodyHybridMode:
    """Hybrid search mode."""

    def test_hybrid_mode_has_both_components(self) -> None:
        body = build_search_body("hybrid query", search_mode="hybrid", top_k=10)
        assert "find_text" in body
        assert "vector_search" in body
        assert body["find_text"]["query"] == "hybrid query"
        assert body["vector_search"]["query"] == "hybrid query"

    def test_hybrid_config_included(self) -> None:
        config = {"vector_weight": 0.6, "text_weight": 0.4, "rerank_results": True}
        body = build_search_body("query", search_mode="hybrid", hybrid_config=config)
        assert body["hybrid_config"] == config

    def test_hybrid_config_ignored_for_other_modes(self) -> None:
        config = {"vector_weight": 0.6, "text_weight": 0.4}
        body = build_search_body("query", search_mode="text", hybrid_config=config)
        assert "hybrid_config" not in body


class TestBuildSearchBodyVectorBoostedMode:
    """Vector-boosted search mode."""

    def test_vector_boosted_has_both_components(self) -> None:
        body = build_search_body("boosted query", search_mode="vector_boosted", top_k=10)
        assert "find_text" in body
        assert "vector_search" in body

    def test_vector_boost_config_included(self) -> None:
        config = {"boost_factor": 1.3, "similarity_threshold": 0.75}
        body = build_search_body("query", search_mode="vector_boosted", vector_boost_config=config)
        assert body["vector_boost_config"] == config

    def test_vector_boost_config_ignored_for_other_modes(self) -> None:
        config = {"boost_factor": 1.3}
        body = build_search_body("query", search_mode="hybrid", vector_boost_config=config)
        assert "vector_boost_config" not in body


class TestBuildSearchBodyFiltersAndSort:
    """Filtering, sorting, and extra kwargs."""

    def test_where_filter(self) -> None:
        where = {"$": {"all_of": [{"status__eq": "published"}]}}
        body = build_search_body("query", where=where)
        assert body["where"] == where

    def test_sort(self) -> None:
        body = build_search_body("query", sort=["-_sys.created_at", "title"])
        assert body["sort"] == ["-_sys.created_at", "title"]

    def test_search_kwargs_override(self) -> None:
        body = build_search_body(
            "query",
            search_mode="hybrid",
            top_k=5,
            search_kwargs={"limit": 50, "ignore_unknown_fields": True},
        )
        # search_kwargs overrides limit
        assert body["limit"] == 50
        assert body["ignore_unknown_fields"] is True

    def test_no_optional_params(self) -> None:
        body = build_search_body("query", search_mode="text", top_k=5)
        assert "where" not in body
        assert "sort" not in body
        assert "hybrid_config" not in body
        assert "vector_boost_config" not in body

    def test_search_kwargs_limit_does_not_update_vector_top_k(self) -> None:
        body = build_search_body(
            "query",
            search_mode="hybrid",
            top_k=5,
            search_kwargs={"limit": 50},
        )
        assert body["limit"] == 50
        assert body["vector_search"]["top_k"] == 5

    def test_defaults(self) -> None:
        body = build_search_body("query")
        assert body["search_mode"] == "hybrid"
        assert body["limit"] == 5

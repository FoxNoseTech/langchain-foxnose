"""Pure function to build search request bodies for the FoxNose Flux _search endpoint.

.. deprecated:: 0.3.0
    The retriever now uses SDK convenience methods directly.
    This module will be removed in v0.4.0.
"""

from __future__ import annotations

import warnings
from typing import Any


def build_search_body(
    query: str,
    *,
    search_mode: str = "hybrid",
    top_k: int = 5,
    search_fields: list[str] | None = None,
    text_threshold: float | None = None,
    vector_fields: list[str] | None = None,
    similarity_threshold: float | None = None,
    where: dict[str, Any] | None = None,
    hybrid_config: dict[str, Any] | None = None,
    vector_boost_config: dict[str, Any] | None = None,
    sort: list[str] | None = None,
    search_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a search request body for the FoxNose Flux ``_search`` endpoint.

    .. deprecated:: 0.3.0
        This function is deprecated and will be removed in v0.4.0.
        The retriever now uses SDK convenience methods directly.

    This is a pure function with no side effects, making it easy to test
    independently of any client or retriever logic.

    Args:
        query: The user's search query text.
        search_mode: One of ``"text"``, ``"vector"``, ``"hybrid"``, or
            ``"vector_boosted"``.
        top_k: Maximum number of results to return.
        search_fields: Fields for text search (``find_text.fields``).
        text_threshold: Typo tolerance for text search (``find_text.threshold``).
        vector_fields: Fields for vector search (``vector_search.fields``).
        similarity_threshold: Minimum cosine similarity (``vector_search.similarity_threshold``).
        where: Structured filter object.
        hybrid_config: Hybrid mode weight/rerank configuration.
        vector_boost_config: Vector-boosted mode configuration.
        sort: Sort fields (prefix with ``-`` for descending).
        search_kwargs: Extra parameters merged into the body last (overrides).
            Note: overriding ``"limit"`` here will **not** update
            ``vector_search.top_k`` — only the outer limit changes.

    Returns:
        A dictionary suitable for passing as ``body`` to
        ``FluxClient.search()`` / ``AsyncFluxClient.search()``.
    """
    warnings.warn(
        "build_search_body() is deprecated and will be removed in v0.4.0. "
        "The retriever now uses SDK convenience methods directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    body: dict[str, Any] = {
        "search_mode": search_mode,
        "limit": top_k,
    }

    # Text search component
    if search_mode in ("text", "hybrid", "vector_boosted"):
        find_text: dict[str, Any] = {"query": query}
        if search_fields is not None:
            find_text["fields"] = search_fields
        if text_threshold is not None:
            find_text["threshold"] = text_threshold
        body["find_text"] = find_text

    # Vector search component
    if search_mode in ("vector", "hybrid", "vector_boosted"):
        vector_search: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
        }
        if vector_fields is not None:
            vector_search["fields"] = vector_fields
        if similarity_threshold is not None:
            vector_search["similarity_threshold"] = similarity_threshold
        body["vector_search"] = vector_search

    # Mode-specific configs
    if hybrid_config is not None and search_mode == "hybrid":
        body["hybrid_config"] = hybrid_config

    if vector_boost_config is not None and search_mode == "vector_boosted":
        body["vector_boost_config"] = vector_boost_config

    # Filtering & sorting
    if where is not None:
        body["where"] = where

    if sort is not None:
        body["sort"] = sort

    # Extra kwargs merged last (can override anything)
    if search_kwargs:
        body.update(search_kwargs)

    return body

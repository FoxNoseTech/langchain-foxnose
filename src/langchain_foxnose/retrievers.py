"""FoxNose retriever for LangChain."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, model_validator

from langchain_foxnose._document_mapper import map_results_to_documents
from langchain_foxnose._search import build_search_body

try:
    from foxnose_sdk.flux import AsyncFluxClient, FluxClient
except ImportError:  # pragma: no cover
    FluxClient = None  # type: ignore[assignment,misc]
    AsyncFluxClient = None  # type: ignore[assignment,misc]


class FoxNoseRetriever(BaseRetriever):
    """LangChain retriever backed by FoxNose Flux search.

    Uses the FoxNose ``_search`` endpoint to retrieve documents with support
    for text, vector, hybrid, and vector-boosted search modes.

    Example:
        .. code-block:: python

            from foxnose_sdk.flux import FluxClient
            from foxnose_sdk.auth import SimpleKeyAuth
            from langchain_foxnose import FoxNoseRetriever

            client = FluxClient(
                base_url="https://<env_key>.fxns.io",
                api_prefix="my_api",
                auth=SimpleKeyAuth("pk", "sk"),
            )
            retriever = FoxNoseRetriever(
                client=client,
                folder_path="knowledge-base",
                page_content_field="body",
                search_mode="hybrid",
                top_k=5,
            )
            docs = retriever.invoke("How do I reset my password?")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Client injection ---
    client: Any | None = None
    """Synchronous :class:`~foxnose_sdk.flux.FluxClient` instance."""

    async_client: Any | None = None
    """Asynchronous :class:`~foxnose_sdk.flux.AsyncFluxClient` instance."""

    # --- Required ---
    folder_path: str
    """Folder path in FoxNose (e.g. ``"knowledge-base"``)."""

    # --- Content mapping (exactly one required) ---
    page_content_field: str | None = None
    """Single ``data`` field whose value becomes ``page_content``."""

    page_content_fields: list[str] | None = None
    """Multiple ``data`` fields concatenated into ``page_content``."""

    page_content_separator: str = "\n\n"
    """Separator used when concatenating *page_content_fields*."""

    page_content_mapper: Callable[[dict[str, Any]], str] | None = None
    """Custom callable ``(result_dict) -> str`` for full control over
    ``page_content`` extraction."""

    # --- Metadata ---
    metadata_fields: list[str] | None = None
    """Whitelist of ``data`` fields to include in document metadata.
    Mutually exclusive with ``exclude_metadata_fields``."""

    exclude_metadata_fields: list[str] | None = None
    """Blacklist of ``data`` fields to exclude from document metadata.
    Mutually exclusive with ``metadata_fields``."""

    include_sys_metadata: bool = True
    """Whether to include ``_sys`` fields (key, folder, created_at, updated_at)
    in document metadata."""

    # --- Search configuration ---
    search_mode: str = "hybrid"
    """Search mode: ``"text"``, ``"vector"``, ``"hybrid"``, or ``"vector_boosted"``."""

    search_fields: list[str] | None = None
    """Fields for text search (``find_text.fields``)."""

    text_threshold: float | None = None
    """Typo-tolerance threshold for text search (``find_text.threshold``, 0-1)."""

    vector_fields: list[str] | None = None
    """Fields for vector search (``vector_search.fields``)."""

    similarity_threshold: float | None = None
    """Minimum cosine similarity for vector search (0-1)."""

    top_k: int = 5
    """Maximum number of results to return. Must be >= 1."""

    where: dict[str, Any] | None = None
    """Persistent structured filter applied to every search."""

    hybrid_config: dict[str, Any] | None = None
    """Hybrid mode configuration (``vector_weight``, ``text_weight``, ``rerank_results``)."""

    vector_boost_config: dict[str, Any] | None = None
    """Vector-boosted mode configuration (``boost_factor``, ``similarity_threshold``,
    ``max_boost_results``)."""

    sort: list[str] | None = None
    """Sort fields (prefix with ``-`` for descending)."""

    search_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra parameters merged into the search body (overrides other settings).

    .. note::

        Overriding ``"limit"`` here does **not** update ``vector_search.top_k``.
        Only the outer ``limit`` changes.
    """

    @model_validator(mode="after")
    def _validate_config(self) -> FoxNoseRetriever:
        # At least one client
        if self.client is None and self.async_client is None:
            raise ValueError(
                "At least one of 'client' (FluxClient) or "
                "'async_client' (AsyncFluxClient) must be provided."
            )

        # Exactly one content mapping strategy
        strategies = [
            self.page_content_field is not None,
            self.page_content_fields is not None,
            self.page_content_mapper is not None,
        ]
        if sum(strategies) == 0:
            raise ValueError(
                "Exactly one content mapping strategy is required: "
                "'page_content_field', 'page_content_fields', or 'page_content_mapper'."
            )
        if sum(strategies) > 1:
            raise ValueError(
                "Only one content mapping strategy may be set. "
                "Choose one of: 'page_content_field', 'page_content_fields', "
                "or 'page_content_mapper'."
            )

        # Valid search mode
        valid_modes = {"text", "vector", "hybrid", "vector_boosted"}
        if self.search_mode not in valid_modes:
            raise ValueError(
                f"Invalid search_mode '{self.search_mode}'. "
                f"Must be one of: {', '.join(sorted(valid_modes))}."
            )

        # top_k must be positive
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}.")

        # page_content_fields must be non-empty when provided
        if self.page_content_fields is not None and len(self.page_content_fields) == 0:
            raise ValueError("'page_content_fields' must not be empty.")

        # metadata_fields and exclude_metadata_fields are mutually exclusive
        if self.metadata_fields is not None and self.exclude_metadata_fields is not None:
            raise ValueError(
                "'metadata_fields' and 'exclude_metadata_fields' are mutually exclusive. "
                "Set only one."
            )

        # Threshold range checks
        if self.text_threshold is not None and not (0 <= self.text_threshold <= 1):
            raise ValueError(
                f"text_threshold must be between 0 and 1, got {self.text_threshold}."
            )
        if self.similarity_threshold is not None and not (0 <= self.similarity_threshold <= 1):
            raise ValueError(
                f"similarity_threshold must be between 0 and 1, got {self.similarity_threshold}."
            )

        # search_kwargs must not override search_mode (creates inconsistent body)
        if "search_mode" in self.search_kwargs:
            raise ValueError(
                "Do not override 'search_mode' via 'search_kwargs'. "
                "Set 'search_mode' directly instead."
            )

        return self

    @classmethod
    def from_client_params(
        cls,
        *,
        base_url: str,
        api_prefix: str,
        auth: Any,
        folder_path: str,
        async_mode: bool = False,
        timeout: float = 15.0,
        **kwargs: Any,
    ) -> FoxNoseRetriever:
        """Create a retriever by constructing the Flux client internally.

        Args:
            base_url: FoxNose environment URL (e.g. ``"https://<env_key>.fxns.io"``).
            api_prefix: Flux API prefix.
            auth: An :class:`~foxnose_sdk.auth.AuthStrategy` instance.
            folder_path: Folder path to search.
            async_mode: If ``True``, create an ``AsyncFluxClient`` instead.
            timeout: HTTP timeout in seconds.
            **kwargs: Additional arguments passed to :class:`FoxNoseRetriever`.

        Returns:
            A configured :class:`FoxNoseRetriever` instance.
        """
        from foxnose_sdk.flux import AsyncFluxClient as _AsyncFluxClient
        from foxnose_sdk.flux import FluxClient as _FluxClient

        if async_mode:
            ac = _AsyncFluxClient(
                base_url=base_url,
                api_prefix=api_prefix,
                auth=auth,
                timeout=timeout,
            )
            return cls(async_client=ac, folder_path=folder_path, **kwargs)
        else:
            c = _FluxClient(
                base_url=base_url,
                api_prefix=api_prefix,
                auth=auth,
                timeout=timeout,
            )
            return cls(client=c, folder_path=folder_path, **kwargs)

    def _build_body(self, query: str) -> dict[str, Any]:
        """Build the search request body."""
        return build_search_body(
            query,
            search_mode=self.search_mode,
            top_k=self.top_k,
            search_fields=self.search_fields,
            text_threshold=self.text_threshold,
            vector_fields=self.vector_fields,
            similarity_threshold=self.similarity_threshold,
            where=self.where,
            hybrid_config=self.hybrid_config,
            vector_boost_config=self.vector_boost_config,
            sort=self.sort,
            search_kwargs=self.search_kwargs,
        )

    def _map_results(self, results: list[dict[str, Any]]) -> list[Document]:
        """Map raw FoxNose results to LangChain Documents."""
        return map_results_to_documents(
            results,
            page_content_field=self.page_content_field,
            page_content_fields=self.page_content_fields,
            page_content_separator=self.page_content_separator,
            page_content_mapper=self.page_content_mapper,
            metadata_fields=self.metadata_fields,
            exclude_metadata_fields=self.exclude_metadata_fields,
            include_sys_metadata=self.include_sys_metadata,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Synchronous retrieval using FluxClient."""
        if self.client is None:
            raise ValueError(
                "Synchronous retrieval requires a 'client' (FluxClient). "
                "Either provide a 'client' or use 'ainvoke()' with an 'async_client'."
            )
        body = self._build_body(query)
        response = self.client.search(self.folder_path, body=body)
        return self._map_results(response.get("results", []))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any,
    ) -> list[Document]:
        """Async retrieval using AsyncFluxClient when available.

        Falls back to the default ``run_in_executor`` behaviour provided by
        :class:`~langchain_core.retrievers.BaseRetriever` when only a sync
        client is available.
        """
        if self.async_client is None:
            # Fall back to sync-in-executor (BaseRetriever default)
            return await super()._aget_relevant_documents(query, run_manager=run_manager)
        body = self._build_body(query)
        response = await self.async_client.search(self.folder_path, body=body)
        return self._map_results(response.get("results", []))

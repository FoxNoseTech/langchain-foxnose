"""FoxNose retriever for LangChain."""

from __future__ import annotations

import math
from typing import Any, Callable

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, model_validator

from langchain_foxnose._document_mapper import map_results_to_documents
from langchain_foxnose._validators import (
    StrictHybridConfig,
    StrictVectorBoostConfig,
    split_search_kwargs,
    validate_search_kwargs,
)

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
    """Fields for vector search with auto-generated embeddings
    (``vector_search.fields``).  Mutually exclusive with ``vector_field``."""

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
    """Extra parameters merged into the search request.

    Known keys like ``limit`` and ``offset`` are extracted as named
    parameters; the rest are passed through ``**extra_body`` to the
    SDK convenience methods.

    .. note::

        Keys that conflict with ``SearchRequest`` fields (e.g.
        ``"search_mode"``, ``"vector_search"``) are rejected at
        validation time.
    """

    # --- Custom embeddings (vector_field_search) ---
    embeddings: Embeddings | None = None
    """Optional LangChain Embeddings model.  When set together with
    ``vector_field``, the retriever converts the query text into a vector
    at query time via ``embeddings.embed_query()``.

    .. warning::

        The query text may be sent to a third-party embedding provider
        (e.g. OpenAI) depending on the Embeddings implementation.
    """

    query_vector: list[float] | None = Field(default=None, repr=False)
    """Pre-computed query vector for ``vector_field_search``.  When set
    together with ``vector_field``, this static vector is used on every
    search invocation.  Mutually exclusive with ``embeddings``."""

    vector_field: str | None = None
    """Single field name for custom-embedding vector search
    (``vector_field_search``).  Required when ``embeddings`` or
    ``query_vector`` is provided.  Mutually exclusive with ``vector_fields``."""

    @model_validator(mode="before")
    @classmethod
    def _alias_k_to_top_k(cls, data: Any) -> Any:
        """Accept ``k`` as a constructor alias for ``top_k``."""
        if isinstance(data, dict):
            k_val = data.pop("k", None)
            if k_val is not None:
                if "top_k" in data:
                    raise ValueError("Cannot pass both 'top_k' and 'k'. Use one or the other.")
                data["top_k"] = k_val
        return data

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
            raise ValueError(f"text_threshold must be between 0 and 1, got {self.text_threshold}.")
        if self.similarity_threshold is not None and not (0 <= self.similarity_threshold <= 1):
            raise ValueError(
                f"similarity_threshold must be between 0 and 1, got {self.similarity_threshold}."
            )

        # search_kwargs must not contain conflicting keys
        validate_search_kwargs(self.search_kwargs)

        # --- Custom embedding validation ---
        has_embeddings = self.embeddings is not None
        has_query_vector = self.query_vector is not None
        has_vector_field = self.vector_field is not None

        # embeddings and query_vector are mutually exclusive
        if has_embeddings and has_query_vector:
            raise ValueError(
                "'embeddings' and 'query_vector' are mutually exclusive. Provide only one."
            )

        # If embeddings or query_vector is set, vector_field is required
        if (has_embeddings or has_query_vector) and not has_vector_field:
            raise ValueError(
                "'vector_field' is required when 'embeddings' or 'query_vector' is set."
            )

        # vector_field without a source is invalid
        if has_vector_field and not has_embeddings and not has_query_vector:
            raise ValueError("'vector_field' requires either 'embeddings' or 'query_vector'.")

        # Custom embeddings only valid in vector / vector_boosted modes
        if (has_embeddings or has_query_vector or has_vector_field) and self.search_mode not in (
            "vector",
            "vector_boosted",
        ):
            raise ValueError(
                f"'embeddings', 'query_vector', and 'vector_field' are only "
                f"supported in 'vector' and 'vector_boosted' search modes, "
                f"got '{self.search_mode}'."
            )

        # vector_field and vector_fields are mutually exclusive
        if has_vector_field and self.vector_fields is not None:
            raise ValueError(
                "'vector_field' and 'vector_fields' are mutually exclusive. "
                "'vector_field' is for custom-embedding search, "
                "'vector_fields' is for auto-generated embedding search."
            )

        # query_vector must be non-empty with finite values
        if has_query_vector:
            if len(self.query_vector) == 0:  # type: ignore[arg-type]
                raise ValueError("'query_vector' must not be empty.")
            if not all(math.isfinite(v) for v in self.query_vector):  # type: ignore[union-attr]
                raise ValueError("All values in 'query_vector' must be finite (no NaN/Inf).")

        # Validate config dicts through strict models (fail-fast on unknown keys)
        if self.hybrid_config is not None:
            StrictHybridConfig(**self.hybrid_config)
        if self.vector_boost_config is not None:
            StrictVectorBoostConfig(**self.vector_boost_config)

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

    # --- Internal helpers ---

    def _build_find_text(self, query: str) -> dict[str, Any]:
        """Build the ``find_text`` dict for text / hybrid / boosted modes."""
        find_text: dict[str, Any] = {"query": query}
        if self.search_fields is not None:
            find_text["fields"] = self.search_fields
        if self.text_threshold is not None:
            find_text["threshold"] = self.text_threshold
        return find_text

    def _build_extra_body(self) -> dict[str, Any]:
        """Build ``**extra_body`` kwargs from where, sort, and search_kwargs."""
        _named, extra = split_search_kwargs(self.search_kwargs)
        if self.where is not None:
            extra.setdefault("where", self.where)
        if self.sort is not None:
            extra.setdefault("sort", self.sort)
        return extra

    def _get_named_overrides(self) -> dict[str, Any]:
        """Extract named parameter overrides from search_kwargs."""
        named, _extra = split_search_kwargs(self.search_kwargs)
        return named

    def _resolve_query_vector(self, query: str) -> list[float]:
        """Resolve the query vector for vector_field mode (sync)."""
        if self.query_vector is not None:
            return self.query_vector
        if self.embeddings is not None:
            return self.embeddings.embed_query(query)
        raise ValueError(  # pragma: no cover — guarded by validator
            "vector_field mode requires 'embeddings' or 'query_vector'."
        )

    async def _aresolve_query_vector(self, query: str) -> list[float]:
        """Resolve the query vector for vector_field mode (async)."""
        if self.query_vector is not None:
            return self.query_vector
        if self.embeddings is not None:
            return await self.embeddings.aembed_query(query)
        raise ValueError(  # pragma: no cover — guarded by validator
            "vector_field mode requires 'embeddings' or 'query_vector'."
        )

    def _effective_top_k(self, override: int | None = None) -> int:
        """Return the effective top_k, preferring a runtime override."""
        if override is not None:
            if not isinstance(override, int) or override < 1:
                raise ValueError(f"Runtime top_k must be an integer >= 1, got {override!r}.")
            return override
        return self.top_k

    def _execute_search(
        self, client: Any, query: str, *, top_k: int | None = None
    ) -> dict[str, Any]:
        """Dispatch to the appropriate SDK convenience method (sync)."""
        extra = self._build_extra_body()
        named = self._get_named_overrides()
        effective_top_k = self._effective_top_k(top_k)

        if self.search_mode == "text":
            return self._search_text(client, query, named, extra, effective_top_k)
        elif self.search_mode == "vector":
            return self._search_vector(client, query, named, extra, effective_top_k)
        elif self.search_mode == "hybrid":
            return self._search_hybrid(client, query, named, extra, effective_top_k)
        elif self.search_mode == "vector_boosted":
            return self._search_boosted(client, query, named, extra, effective_top_k)
        else:  # pragma: no cover — guarded by validator
            raise ValueError(f"Unknown search_mode: {self.search_mode}")

    async def _aexecute_search(
        self, client: Any, query: str, *, top_k: int | None = None
    ) -> dict[str, Any]:
        """Dispatch to the appropriate SDK convenience method (async)."""
        extra = self._build_extra_body()
        named = self._get_named_overrides()
        effective_top_k = self._effective_top_k(top_k)

        if self.search_mode == "text":
            return await self._asearch_text(client, query, named, extra, effective_top_k)
        elif self.search_mode == "vector":
            return await self._asearch_vector(client, query, named, extra, effective_top_k)
        elif self.search_mode == "hybrid":
            return await self._asearch_hybrid(client, query, named, extra, effective_top_k)
        elif self.search_mode == "vector_boosted":
            return await self._asearch_boosted(client, query, named, extra, effective_top_k)
        else:  # pragma: no cover
            raise ValueError(f"Unknown search_mode: {self.search_mode}")

    # --- Per-mode dispatch (sync) ---

    def _search_text(
        self, client: Any, query: str, named: dict, extra: dict, top_k: int
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "search_mode": "text",
            "find_text": self._build_find_text(query),
            "limit": named.get("limit", top_k),
        }
        if "offset" in named:
            body["offset"] = named["offset"]
        if self.where is not None:
            body["where"] = self.where
        if self.sort is not None:
            body["sort"] = self.sort
        # Merge extra (may override instance-level where/sort from search_kwargs)
        body.update(extra)
        return client.search(self.folder_path, body=body)

    def _search_vector(
        self, client: Any, query: str, named: dict, extra: dict, top_k: int
    ) -> dict[str, Any]:
        if self.vector_field is not None:
            qv = self._resolve_query_vector(query)
            return client.vector_field_search(
                self.folder_path,
                field=self.vector_field,
                query_vector=qv,
                top_k=top_k,
                similarity_threshold=self.similarity_threshold,
                limit=named.get("limit", top_k),
                offset=named.get("offset"),
                **extra,
            )
        return client.vector_search(
            self.folder_path,
            query=query,
            fields=self.vector_fields,
            top_k=top_k,
            similarity_threshold=self.similarity_threshold,
            limit=named.get("limit", top_k),
            offset=named.get("offset"),
            **extra,
        )

    def _search_hybrid(
        self, client: Any, query: str, named: dict, extra: dict, top_k: int
    ) -> dict[str, Any]:
        hc = StrictHybridConfig(**(self.hybrid_config or {}))
        return client.hybrid_search(
            self.folder_path,
            query=query,
            find_text=self._build_find_text(query),
            fields=self.vector_fields,
            top_k=top_k,
            similarity_threshold=self.similarity_threshold,
            vector_weight=hc.vector_weight,
            text_weight=hc.text_weight,
            rerank_results=hc.rerank_results,
            limit=named.get("limit"),
            offset=named.get("offset"),
            **extra,
        )

    def _search_boosted(
        self, client: Any, query: str, named: dict, extra: dict, top_k: int
    ) -> dict[str, Any]:
        bc = StrictVectorBoostConfig(**(self.vector_boost_config or {}))
        kwargs: dict[str, Any] = {
            "find_text": self._build_find_text(query),
            "top_k": top_k,
            "similarity_threshold": self.similarity_threshold,
            "boost_factor": bc.boost_factor,
            "boost_similarity_threshold": bc.similarity_threshold,
            "max_boost_results": bc.max_boost_results,
            "limit": named.get("limit"),
            "offset": named.get("offset"),
        }
        if self.vector_field is not None:
            qv = self._resolve_query_vector(query)
            kwargs["field"] = self.vector_field
            kwargs["query_vector"] = qv
        else:
            kwargs["query"] = query
        return client.boosted_search(self.folder_path, **kwargs, **extra)

    # --- Per-mode dispatch (async) ---

    async def _asearch_text(
        self, client: Any, query: str, named: dict, extra: dict, top_k: int
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "search_mode": "text",
            "find_text": self._build_find_text(query),
            "limit": named.get("limit", top_k),
        }
        if "offset" in named:
            body["offset"] = named["offset"]
        if self.where is not None:
            body["where"] = self.where
        if self.sort is not None:
            body["sort"] = self.sort
        # Merge extra (may override instance-level where/sort from search_kwargs)
        body.update(extra)
        return await client.search(self.folder_path, body=body)

    async def _asearch_vector(
        self, client: Any, query: str, named: dict, extra: dict, top_k: int
    ) -> dict[str, Any]:
        if self.vector_field is not None:
            qv = await self._aresolve_query_vector(query)
            return await client.vector_field_search(
                self.folder_path,
                field=self.vector_field,
                query_vector=qv,
                top_k=top_k,
                similarity_threshold=self.similarity_threshold,
                limit=named.get("limit", top_k),
                offset=named.get("offset"),
                **extra,
            )
        return await client.vector_search(
            self.folder_path,
            query=query,
            fields=self.vector_fields,
            top_k=top_k,
            similarity_threshold=self.similarity_threshold,
            limit=named.get("limit", top_k),
            offset=named.get("offset"),
            **extra,
        )

    async def _asearch_hybrid(
        self, client: Any, query: str, named: dict, extra: dict, top_k: int
    ) -> dict[str, Any]:
        hc = StrictHybridConfig(**(self.hybrid_config or {}))
        return await client.hybrid_search(
            self.folder_path,
            query=query,
            find_text=self._build_find_text(query),
            fields=self.vector_fields,
            top_k=top_k,
            similarity_threshold=self.similarity_threshold,
            vector_weight=hc.vector_weight,
            text_weight=hc.text_weight,
            rerank_results=hc.rerank_results,
            limit=named.get("limit"),
            offset=named.get("offset"),
            **extra,
        )

    async def _asearch_boosted(
        self, client: Any, query: str, named: dict, extra: dict, top_k: int
    ) -> dict[str, Any]:
        bc = StrictVectorBoostConfig(**(self.vector_boost_config or {}))
        kwargs: dict[str, Any] = {
            "find_text": self._build_find_text(query),
            "top_k": top_k,
            "similarity_threshold": self.similarity_threshold,
            "boost_factor": bc.boost_factor,
            "boost_similarity_threshold": bc.similarity_threshold,
            "max_boost_results": bc.max_boost_results,
            "limit": named.get("limit"),
            "offset": named.get("offset"),
        }
        if self.vector_field is not None:
            qv = await self._aresolve_query_vector(query)
            kwargs["field"] = self.vector_field
            kwargs["query_vector"] = qv
        else:
            kwargs["query"] = query
        return await client.boosted_search(self.folder_path, **kwargs, **extra)

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
        **kwargs: Any,
    ) -> list[Document]:
        """Synchronous retrieval using FluxClient."""
        if self.client is None:
            raise ValueError(
                "Synchronous retrieval requires a 'client' (FluxClient). "
                "Either provide a 'client' or use 'ainvoke()' with an 'async_client'."
            )
        top_k_override = self._resolve_top_k_kwarg(kwargs)
        response = self._execute_search(self.client, query, top_k=top_k_override)
        docs = self._map_results(response.get("results", []))
        if top_k_override is not None:
            docs = docs[:top_k_override]
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any,
        **kwargs: Any,
    ) -> list[Document]:
        """Async retrieval using AsyncFluxClient when available.

        Falls back to running the sync method via ``run_in_executor`` when
        only a sync client is available.  This preserves runtime kwargs
        (e.g. ``top_k``) which ``super()._aget_relevant_documents`` does not
        forward.
        """
        top_k_override = self._resolve_top_k_kwarg(kwargs)
        if self.async_client is None:
            import functools

            from langchain_core.runnables.config import run_in_executor

            return await run_in_executor(
                None,
                functools.partial(
                    self._get_relevant_documents,
                    query,
                    run_manager=run_manager.get_sync(),
                    **kwargs,
                ),
            )
        response = await self._aexecute_search(self.async_client, query, top_k=top_k_override)
        docs = self._map_results(response.get("results", []))
        if top_k_override is not None:
            docs = docs[:top_k_override]
        return docs

    @staticmethod
    def _resolve_top_k_kwarg(kwargs: dict[str, Any]) -> int | None:
        """Extract ``top_k`` from kwargs, accepting ``k`` as an alias."""
        top_k = kwargs.get("top_k")
        k = kwargs.get("k")
        if top_k is not None and k is not None:
            raise ValueError("Cannot pass both 'top_k' and 'k'. Use one or the other.")
        return top_k if top_k is not None else k

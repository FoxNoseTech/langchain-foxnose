"""FoxNose document loader for LangChain."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any
from urllib.parse import parse_qs, urlparse

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_foxnose._deprecation import warn_deprecated_field
from langchain_foxnose._document_mapper import map_results_to_documents

try:
    from foxnose_sdk.flux import AsyncFluxClient, FluxClient
except ImportError:  # pragma: no cover
    FluxClient = None  # type: ignore[assignment,misc]
    AsyncFluxClient = None  # type: ignore[assignment,misc]


def _extract_cursor(next_value: Any) -> str | None:
    """Normalise a ``next`` field into the token ``list_resources`` expects.

    FoxNose returns ``next`` as a FULL URL, e.g.
    ``http://host/api/articles?limit=2&next=9avd3azzc0tp`` -- not as the opaque
    token the ``next`` query parameter takes. Feeding the whole URL back means
    the backend cannot parse it, silently answers with page one again, and
    returns the same ``next``: an infinite loop that re-fetches the first page
    forever. A plain token is passed through unchanged, so a backend that
    returns one keeps working.
    """
    if not next_value or not isinstance(next_value, str):
        return None
    if "://" not in next_value:
        return next_value
    values = parse_qs(urlparse(next_value).query).get("next")
    return values[0] if values else None


class FoxNoseLoader(BaseLoader):
    """LangChain document loader backed by FoxNose Flux ``list_resources``.

    Iterates over all resources in a FoxNose collection with automatic
    cursor-based pagination.

    Example:
        .. code-block:: python

            from foxnose_sdk.flux import FluxClient
            from foxnose_sdk.auth import SimpleKeyAuth
            from langchain_foxnose import FoxNoseLoader

            client = FluxClient(
                base_url="https://<env_key>.fxns.io",
                api_prefix="my_api",
                auth=SimpleKeyAuth("pk", "sk"),
            )
            loader = FoxNoseLoader(
                client=client,
                collection_path="knowledge-base",
                page_content_field="body",
            )
            docs = loader.load()

    Args:
        client: Synchronous :class:`~foxnose_sdk.flux.FluxClient` instance.
        async_client: Asynchronous :class:`~foxnose_sdk.flux.AsyncFluxClient` instance.
        collection_path: Collection path in FoxNose (e.g. ``"knowledge-base"``).
            Renamed from ``folder_path`` in 0.4.0; the legacy kwarg is still
            accepted with a ``DeprecationWarning`` and will be removed in 1.0.
        page_content_field: Single ``data`` field whose value becomes ``page_content``.
        page_content_fields: Multiple ``data`` fields concatenated into ``page_content``.
        page_content_separator: Separator when using *page_content_fields*.
        page_content_mapper: Custom callable ``(result_dict) -> str`` for full control.
        metadata_fields: Whitelist of ``data`` fields to include in metadata.
        exclude_metadata_fields: Blacklist of ``data`` fields to exclude from metadata.
        include_sys_metadata: Whether to include ``_sys`` fields in metadata.
        params: Query parameters forwarded to ``list_resources``.
        truncate_text: Cap the length of every ``text``-typed field in the
            response, in characters. Sent as the ``truncate_text`` query
            parameter. Must be >= 1. Prefer this over a ``truncate_text`` key
            inside *params*; setting both raises.
        batch_size: Page size for ``list_resources`` calls (must be >= 1).
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        async_client: Any | None = None,
        collection_path: str | None = None,
        folder_path: str | None = None,  # deprecated alias for collection_path
        page_content_field: str | None = None,
        page_content_fields: list[str] | None = None,
        page_content_separator: str = "\n\n",
        page_content_mapper: Callable[[dict[str, Any]], str] | None = None,
        metadata_fields: list[str] | None = None,
        exclude_metadata_fields: list[str] | None = None,
        include_sys_metadata: bool = True,
        params: dict[str, Any] | None = None,
        truncate_text: int | None = None,
        batch_size: int = 100,
    ) -> None:
        # --- folder_path → collection_path migration (FOX-M0-01) ---
        if folder_path is not None and collection_path is not None:
            raise ValueError("Pass either folder_path (deprecated) or collection_path, not both.")
        if folder_path is not None:
            warn_deprecated_field("folder_path", "collection_path")
            collection_path = folder_path
        if collection_path is None:
            raise ValueError("collection_path is required.")
        # --- end migration ---

        # At least one client
        if client is None and async_client is None:
            raise ValueError(
                "At least one of 'client' (FluxClient) or "
                "'async_client' (AsyncFluxClient) must be provided."
            )

        # Exactly one content mapping strategy
        strategies = [
            page_content_field is not None,
            page_content_fields is not None,
            page_content_mapper is not None,
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

        # page_content_fields must be non-empty when provided
        if page_content_fields is not None and len(page_content_fields) == 0:
            raise ValueError("'page_content_fields' must not be empty.")

        # metadata_fields and exclude_metadata_fields are mutually exclusive
        if metadata_fields is not None and exclude_metadata_fields is not None:
            raise ValueError(
                "'metadata_fields' and 'exclude_metadata_fields' are mutually exclusive. "
                "Set only one."
            )

        # batch_size must be positive
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

        # truncate_text must be positive and unambiguous
        if truncate_text is not None:
            if truncate_text < 1:
                raise ValueError(f"truncate_text must be >= 1, got {truncate_text}.")
            if params is not None and "truncate_text" in params:
                raise ValueError(
                    "truncate_text is set both directly and inside params. Set only one."
                )

        self.client = client
        self.async_client = async_client
        self.collection_path = collection_path
        self.page_content_field = page_content_field
        self.page_content_fields = page_content_fields
        self.page_content_separator = page_content_separator
        self.page_content_mapper = page_content_mapper
        self.metadata_fields = metadata_fields
        self.exclude_metadata_fields = exclude_metadata_fields
        self.include_sys_metadata = include_sys_metadata
        self.params: dict[str, Any] = params if params is not None else {}
        self.batch_size = batch_size
        self.truncate_text = truncate_text

    @property
    def folder_path(self) -> str:
        """Deprecated; alias for :attr:`collection_path`. Removed in 1.0."""
        return self.collection_path

    @classmethod
    def from_client_params(
        cls,
        *,
        base_url: str,
        api_prefix: str,
        auth: Any,
        collection_path: str | None = None,
        folder_path: str | None = None,  # deprecated alias for collection_path
        async_mode: bool = False,
        timeout: float = 15.0,
        **kwargs: Any,
    ) -> FoxNoseLoader:
        """Create a loader by constructing the Flux client internally.

        Args:
            base_url: FoxNose environment URL (e.g. ``"https://<env_key>.fxns.io"``).
            api_prefix: Flux API prefix.
            auth: An :class:`~foxnose_sdk.auth.AuthStrategy` instance.
            collection_path: Collection path to load from. Required (or pass the
                deprecated ``folder_path`` alias).
            folder_path: Deprecated alias for ``collection_path``. Emits a
                ``DeprecationWarning`` via ``__init__`` and will be removed in 1.0.
            async_mode: If ``True``, create an ``AsyncFluxClient`` instead.
            timeout: HTTP timeout in seconds.
            **kwargs: Additional arguments passed to :class:`FoxNoseLoader`.

        Returns:
            A configured :class:`FoxNoseLoader` instance.
        """
        from foxnose_sdk.flux import AsyncFluxClient as _AsyncFluxClient
        from foxnose_sdk.flux import FluxClient as _FluxClient

        # Delegate the deprecation warning + both/neither validation to __init__.
        if async_mode:
            ac = _AsyncFluxClient(
                base_url=base_url,
                api_prefix=api_prefix,
                auth=auth,
                timeout=timeout,
            )
            return cls(
                async_client=ac,
                collection_path=collection_path,
                folder_path=folder_path,
                **kwargs,
            )
        c = _FluxClient(
            base_url=base_url,
            api_prefix=api_prefix,
            auth=auth,
            timeout=timeout,
        )
        return cls(
            client=c,
            collection_path=collection_path,
            folder_path=folder_path,
            **kwargs,
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

    def _build_request_params(self, cursor: str | None) -> dict[str, Any]:
        """Build the ``list_resources`` query params for one page."""
        request_params: dict[str, Any] = {**self.params, "limit": self.batch_size}
        if self.truncate_text is not None:
            request_params["truncate_text"] = self.truncate_text
        if cursor is not None:
            request_params["next"] = cursor
        return request_params

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents from FoxNose with cursor-based pagination.

        Yields:
            :class:`langchain_core.documents.Document` objects.

        Raises:
            ValueError: If no synchronous client is available.
        """
        if self.client is None:
            raise ValueError(
                "Synchronous loading requires a 'client' (FluxClient). "
                "Either provide a 'client' or use 'alazy_load()' with an 'async_client'."
            )

        seen_cursors: set[str] = set()
        cursor: str | None = None
        while True:
            request_params = self._build_request_params(cursor)

            response = self.client.list_resources(self.collection_path, params=request_params)
            results = response.get("results", [])
            documents = self._map_results(results)
            yield from documents

            next_cursor = _extract_cursor(response.get("next"))
            # Every cursor is followed at most once. Checking only whether the
            # cursor repeated the previous one would still spin forever on a
            # cycle (A -> B -> A), and would re-yield the repeated page before
            # noticing.
            if next_cursor is None or next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Asynchronously load documents from FoxNose with cursor-based pagination.

        Yields:
            :class:`langchain_core.documents.Document` objects.

        Raises:
            ValueError: If no async client is available.
        """
        if self.async_client is None:
            raise ValueError(
                "Async loading requires an 'async_client' (AsyncFluxClient). "
                "Either provide an 'async_client' or use 'lazy_load()' with a 'client'."
            )

        seen_cursors: set[str] = set()
        cursor: str | None = None
        while True:
            request_params = self._build_request_params(cursor)

            response = await self.async_client.list_resources(
                self.collection_path, params=request_params
            )
            results = response.get("results", [])
            documents = self._map_results(results)
            for doc in documents:
                yield doc

            next_cursor = _extract_cursor(response.get("next"))
            # See lazy_load: a cursor is followed at most once, so a cycle
            # terminates instead of spinning.
            if next_cursor is None or next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor

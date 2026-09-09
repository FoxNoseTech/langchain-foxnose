"""FoxNose document writer for LangChain Documents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.documents import Document

from langchain_foxnose._document_writer import map_document_to_data

try:
    from foxnose_sdk.flux import AsyncFluxClient, FluxClient
except ImportError:  # pragma: no cover
    FluxClient = None  # type: ignore[assignment,misc]
    AsyncFluxClient = None  # type: ignore[assignment,misc]


class FoxNoseBatchWriteError(RuntimeError):
    """Raised when a document batch fails part-way through.

    Flux writes are non-idempotent, are never retried automatically, and cannot
    be rolled back or deleted, so a failed batch leaves earlier documents in
    place. Batches are written strictly sequentially and stop at the first
    failure, which is what makes the three index ranges below meaningful.

    Attributes:
        written_keys: ``resource_key`` of documents ``0 .. failed_index - 1``,
            confirmed written and **not rolled back**.
        failed_index: Index of the document whose write raised. Its outcome is
            **unknown**, not necessarily failed — a
            :class:`~foxnose_sdk.errors.ContentValidationFailed` or
            :class:`~foxnose_sdk.errors.ExternalIdConflict` wrote nothing, but
            an :class:`~foxnose_sdk.errors.UpstreamError` or a transport
            timeout may have written it. Re-read with a GET to find out; do not
            blindly retry.
        pending_indexes: Indexes ``failed_index + 1 .. total - 1``, guaranteed
            **not attempted**.
        total: Number of documents in the batch.
    """

    def __init__(
        self,
        message: str,
        *,
        written_keys: list[str],
        failed_index: int,
        total: int,
    ) -> None:
        super().__init__(message)
        self.written_keys = written_keys
        self.failed_index = failed_index
        self.total = total
        self.pending_indexes = list(range(failed_index + 1, total))

    @property
    def cause(self) -> BaseException | None:
        """The underlying SDK error. Branch on ``type(exc.cause)``.

        A second ``except`` clause after ``FoxNoseBatchWriteError`` never fires
        for a batch write — every underlying error is wrapped.
        """
        return self.__cause__


class FoxNoseWriter:
    """Write LangChain Documents into a FoxNose collection via Flux.

    Each document becomes one FoxNose resource, published immediately.
    This is the write counterpart to :class:`~langchain_foxnose.FoxNoseRetriever`
    and :class:`~langchain_foxnose.FoxNoseLoader`.

    Example:
        .. code-block:: python

            from foxnose_sdk.flux import FluxClient
            from foxnose_sdk.auth import SimpleKeyAuth
            from langchain_core.documents import Document
            from langchain_foxnose import FoxNoseWriter

            client = FluxClient(
                base_url="https://<env_key>.fxns.io",
                api_prefix="my_api",
                auth=SimpleKeyAuth("pk", "sk"),
            )
            writer = FoxNoseWriter(
                client=client,
                collection_path="knowledge-base",
                page_content_field="body",
            )
            keys = writer.add_documents(
                [Document(page_content="Hello", metadata={"title": "Hi"})]
            )

    Args:
        client: Synchronous :class:`~foxnose_sdk.flux.FluxClient` instance.
        async_client: Asynchronous :class:`~foxnose_sdk.flux.AsyncFluxClient` instance.
        collection_path: Collection path in FoxNose (e.g. ``"knowledge-base"``).
        page_content_field: ``data`` field that receives ``page_content``.
        document_mapper: Custom callable ``(document) -> dict`` for full control
            over the written ``data``. Mutually exclusive with
            *page_content_field*.
        metadata_fields: Whitelist of metadata keys to write.
        exclude_metadata_fields: Blacklist of metadata keys to skip.
        include_sys_metadata: Whether to write the read path's ``_sys``-derived
            metadata keys (``key``, ``folder``, ``created_at``, ``updated_at``).
            Defaults to ``False`` — they are not schema fields.
        external_id_key: Metadata key holding an external deduplication id.
            Its value is sent as the resource ``key``. The value must be a
            ``str`` or ``int``; anything else raises ``TypeError``. A document
            without the key is written without one. Reusing a value raises
            :class:`~foxnose_sdk.errors.ExternalIdConflict`.
            The key is kept out of ``data`` when *page_content_field* builds the
            mapping. With a *document_mapper* the mapper decides: whatever it
            returns is written as-is, so a mapper that copies all metadata will
            also write the identifier into ``data``.

    Note:
        Writes require a Flux key with write access. A collection whose
        connection does not accept writes raises
        :class:`~foxnose_sdk.errors.CollectionNotWritable`.

        Data that fails the collection schema comes back as HTTP 422. Match on
        ``status_code``, not on
        :class:`~foxnose_sdk.errors.ContentValidationFailed`: the SDK maps only
        ``content_validation_failed`` onto that class, while a Flux write
        reports ``data_validation_error``, so it arrives as a plain
        :class:`~foxnose_sdk.errors.FoxnoseAPIError`.

    Note:
        Batches are written strictly sequentially and stop at the first
        failure — there is no concurrency option on purpose. Flux writes are
        non-idempotent, are never retried by the SDK, and cannot be deleted, so
        concurrent scheduling would leave the failure report unable to say
        which documents were attempted. See :class:`FoxNoseBatchWriteError`.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        async_client: Any | None = None,
        collection_path: str,
        page_content_field: str | None = None,
        document_mapper: Callable[[Document], dict[str, Any]] | None = None,
        metadata_fields: list[str] | None = None,
        exclude_metadata_fields: list[str] | None = None,
        include_sys_metadata: bool = False,
        external_id_key: str | None = None,
    ) -> None:
        if client is None and async_client is None:
            raise ValueError(
                "At least one of 'client' (FluxClient) or "
                "'async_client' (AsyncFluxClient) must be provided."
            )

        strategies = [page_content_field is not None, document_mapper is not None]
        if sum(strategies) == 0:
            raise ValueError(
                "Exactly one content mapping strategy is required: "
                "'page_content_field' or 'document_mapper'."
            )
        if sum(strategies) > 1:
            raise ValueError(
                "Only one content mapping strategy may be set. "
                "Choose 'page_content_field' or 'document_mapper'."
            )

        if metadata_fields is not None and exclude_metadata_fields is not None:
            raise ValueError(
                "'metadata_fields' and 'exclude_metadata_fields' are mutually "
                "exclusive. Set only one."
            )

        # "" is not a usable metadata key, and allowing it split this field into
        # two readings: _external_id tested `is None` and would have looked up
        # metadata[""], while _map_data tested truthiness and reserved nothing.
        if external_id_key is not None and not external_id_key.strip():
            raise ValueError("'external_id_key' must be a non-empty metadata key name.")

        self.client = client
        self.async_client = async_client
        self.collection_path = collection_path
        self.page_content_field = page_content_field
        self.document_mapper = document_mapper
        self.metadata_fields = metadata_fields
        self.exclude_metadata_fields = exclude_metadata_fields
        self.include_sys_metadata = include_sys_metadata
        self.external_id_key = external_id_key

    @classmethod
    def from_client_params(
        cls,
        *,
        base_url: str,
        api_prefix: str,
        auth: Any,
        collection_path: str,
        async_mode: bool = False,
        timeout: float = 15.0,
        **kwargs: Any,
    ) -> FoxNoseWriter:
        """Create a writer by constructing the Flux client internally.

        Args:
            base_url: FoxNose environment URL (e.g. ``"https://<env_key>.fxns.io"``).
            api_prefix: Flux API prefix.
            auth: An :class:`~foxnose_sdk.auth.AuthStrategy` instance.
            collection_path: Collection path to write to.
            async_mode: If ``True``, create an ``AsyncFluxClient`` instead.
            timeout: HTTP timeout in seconds.
            **kwargs: Additional arguments passed to :class:`FoxNoseWriter`.

        Returns:
            A configured :class:`FoxNoseWriter` instance.
        """
        from foxnose_sdk.flux import AsyncFluxClient as _AsyncFluxClient
        from foxnose_sdk.flux import FluxClient as _FluxClient

        if async_mode:
            return cls(
                async_client=_AsyncFluxClient(
                    base_url=base_url,
                    api_prefix=api_prefix,
                    auth=auth,
                    timeout=timeout,
                ),
                collection_path=collection_path,
                **kwargs,
            )
        return cls(
            client=_FluxClient(
                base_url=base_url,
                api_prefix=api_prefix,
                auth=auth,
                timeout=timeout,
            ),
            collection_path=collection_path,
            **kwargs,
        )

    # --- Internal helpers ---

    def _map_data(self, document: Document) -> dict[str, Any]:
        """Map one Document to the FoxNose ``data`` payload."""
        reserved = () if self.external_id_key is None else (self.external_id_key,)
        return map_document_to_data(
            document,
            page_content_field=self.page_content_field,
            document_mapper=self.document_mapper,
            metadata_fields=self.metadata_fields,
            exclude_metadata_fields=self.exclude_metadata_fields,
            include_sys_metadata=self.include_sys_metadata,
            reserved_metadata_keys=reserved,
        )

    def _external_id(self, document: Document, index: int) -> str | None:
        """Return the external deduplication id for *document*, if configured.

        Raises:
            TypeError: If the metadata value is neither ``str`` nor ``int``.
        """
        if self.external_id_key is None:
            return None
        value = (document.metadata or {}).get(self.external_id_key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TypeError(
                f"Document {index}: metadata['{self.external_id_key}'] must be a "
                f"str or int to be used as an external id, got "
                f"{type(value).__name__}."
            )
        return str(value)

    def _create_kwargs(self, document: Document, index: int) -> dict[str, Any]:
        """Build the optional keyword arguments for ``create_resource``."""
        external_id = self._external_id(document, index)
        return {} if external_id is None else {"key": external_id}

    def _prepare_batch(
        self, documents: Sequence[Document]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Map and validate every document BEFORE any request is sent.

        Local mapping errors (a bad external id, a metadata/content collision,
        a raising custom mapper) must not be able to happen half way through a
        batch: they would surface as a plain ``TypeError``/``ValueError`` with
        no record of what had already been written. Doing all of the pure work
        up front means a local failure guarantees **zero** writes, and
        :class:`FoxNoseBatchWriteError` stays reserved for attempted network
        writes.
        """
        return [
            (self._map_data(document), self._create_kwargs(document, index))
            for index, document in enumerate(documents)
        ]

    @staticmethod
    def _batch_error(
        written_keys: list[str], failed_index: int, total: int
    ) -> FoxNoseBatchWriteError:
        """Build the partial-failure exception for a batch."""
        pending = total - failed_index - 1
        return FoxNoseBatchWriteError(
            f"Failed to write document {failed_index} of {total}. "
            f"{len(written_keys)} document(s) were already written and are not "
            f"rolled back; document {failed_index} has an unknown outcome and "
            f"must be re-read before any retry; {pending} document(s) were not "
            f"attempted.",
            written_keys=written_keys,
            failed_index=failed_index,
            total=total,
        )

    # --- Public API ---

    def add_documents(self, documents: Sequence[Document]) -> list[str]:
        """Create one FoxNose resource per document, sequentially.

        Writes stop at the first failure; documents after it are not attempted.
        All documents are mapped and validated before the first request, so a
        mapping error writes nothing at all.

        Args:
            documents: The documents to write.

        Returns:
            The created ``resource_key`` values, in input order.

        Raises:
            ValueError: If no synchronous client is available, or a document
                cannot be mapped. Raised before any write is attempted.
            TypeError: If a document's external id is not a ``str`` or ``int``.
                Raised before any write is attempted.
            FoxNoseBatchWriteError: If any write fails. Inspect
                ``exc.written_keys``, ``exc.failed_index``,
                ``exc.pending_indexes`` and ``exc.cause``.
        """
        if self.client is None:
            raise ValueError(
                "Synchronous writing requires a 'client' (FluxClient). "
                "Either provide a 'client' or use 'aadd_documents()' with an "
                "'async_client'."
            )
        prepared = self._prepare_batch(documents)
        written: list[str] = []
        total = len(prepared)
        for index, (data, create_kwargs) in enumerate(prepared):
            try:
                response = self.client.create_resource(self.collection_path, data, **create_kwargs)
                # Inside the try on purpose: a malformed success response must
                # not escape as a bare KeyError after a resource was created.
                resource_key = str(response["resource_key"])
            except Exception as exc:
                raise self._batch_error(written, index, total) from exc
            written.append(resource_key)
        return written

    async def aadd_documents(self, documents: Sequence[Document]) -> list[str]:
        """Create one FoxNose resource per document, sequentially.

        Deliberately sequential with an early stop, exactly like
        :meth:`add_documents`: Flux writes are non-idempotent, are never
        retried by the SDK, and cannot be deleted, so overlapping them would
        make :class:`FoxNoseBatchWriteError` unable to say which documents were
        attempted. Chunk the input and call this method per chunk if you need
        concurrency and can accept that ambiguity.

        Args:
            documents: The documents to write.

        Returns:
            The created ``resource_key`` values, in input order.

        Raises:
            ValueError: If no async client is available, or a document cannot
                be mapped. Raised before any write is attempted.
            TypeError: If a document's external id is not a ``str`` or ``int``.
                Raised before any write is attempted.
            FoxNoseBatchWriteError: If any write fails. Inspect
                ``exc.written_keys``, ``exc.failed_index``,
                ``exc.pending_indexes`` and ``exc.cause``.
        """
        if self.async_client is None:
            raise ValueError(
                "Async writing requires an 'async_client' (AsyncFluxClient). "
                "Either provide an 'async_client' or use 'add_documents()' with "
                "a 'client'."
            )
        prepared = self._prepare_batch(documents)
        written: list[str] = []
        total = len(prepared)
        for index, (data, create_kwargs) in enumerate(prepared):
            try:
                response = await self.async_client.create_resource(
                    self.collection_path, data, **create_kwargs
                )
                # Inside the try on purpose: see add_documents.
                resource_key = str(response["resource_key"])
            except Exception as exc:
                raise self._batch_error(written, index, total) from exc
            written.append(resource_key)
        return written

    def update_document(self, resource_key: str, document: Document) -> str:
        """Replace a resource's document and publish a new revision.

        This is a full-document replace, not a partial merge: fields absent
        from the mapped ``data`` are removed from the stored resource.

        Args:
            resource_key: Internal FoxNose resource key (as returned by
                :meth:`add_documents` or found in ``Document.metadata["key"]``).
            document: The replacement document.

        Returns:
            The new ``revision_key``.

        Raises:
            ValueError: If no synchronous client is available.
        """
        if self.client is None:
            raise ValueError(
                "Synchronous writing requires a 'client' (FluxClient). "
                "Either provide a 'client' or use 'aupdate_document()' with an "
                "'async_client'."
            )
        response = self.client.update_resource(
            self.collection_path, resource_key, self._map_data(document)
        )
        return str(response["revision_key"])

    async def aupdate_document(self, resource_key: str, document: Document) -> str:
        """Async variant of :meth:`update_document`.

        Raises:
            ValueError: If no async client is available.
        """
        if self.async_client is None:
            raise ValueError(
                "Async writing requires an 'async_client' (AsyncFluxClient). "
                "Either provide an 'async_client' or use 'update_document()' "
                "with a 'client'."
            )
        response = await self.async_client.update_resource(
            self.collection_path, resource_key, self._map_data(document)
        )
        return str(response["revision_key"])

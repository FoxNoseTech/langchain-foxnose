"""Pure function to map LangChain Documents to FoxNose resource ``data``.

The inverse of :mod:`langchain_foxnose._document_mapper`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.documents import Document

SYS_METADATA_KEYS = frozenset({"key", "folder", "created_at", "updated_at"})
"""Metadata keys injected by the read path from a result's ``_sys`` block.

These are not schema fields, so they are stripped before a write unless the
caller explicitly opts in via ``include_sys_metadata=True``.
"""


def map_document_to_data(
    document: Document,
    *,
    page_content_field: str | None = None,
    document_mapper: Callable[[Document], dict[str, Any]] | None = None,
    metadata_fields: list[str] | None = None,
    exclude_metadata_fields: list[str] | None = None,
    include_sys_metadata: bool = False,
    reserved_metadata_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Convert a LangChain Document into a FoxNose resource ``data`` mapping.

    This is a pure function with no side effects.

    Args:
        document: The document to convert.
        page_content_field: ``data`` field that receives ``page_content``.
            Required unless *document_mapper* is given.
        document_mapper: Custom callable ``(document) -> dict`` for full
            control. When set, EVERY other option is ignored, including
            *reserved_metadata_keys*: the mapper owns its output, and silently
            deleting keys from it would make a mapper that deliberately writes
            an identifier into ``data`` impossible to express.
        metadata_fields: Whitelist of metadata keys to write.
            Mutually exclusive with *exclude_metadata_fields*.
        exclude_metadata_fields: Blacklist of metadata keys to skip.
        include_sys_metadata: Whether to write keys in
            :data:`SYS_METADATA_KEYS`. Defaults to ``False``.
        reserved_metadata_keys: Metadata keys the caller consumes itself
            (e.g. an external-id key). Dropped from the mapping this function
            builds; not applied when *document_mapper* is given.

    Returns:
        A ``data`` mapping suitable for ``FluxClient.create_resource()`` or
        ``update_resource()``.

    Raises:
        ValueError: On conflicting options, a missing *page_content_field*, or
            a metadata key that collides with *page_content_field*.
    """
    if document_mapper is not None:
        # No filtering of any kind on this path -- see *document_mapper* above.
        return dict(document_mapper(document))

    if page_content_field is None:
        raise ValueError("'page_content_field' is required unless 'document_mapper' is provided.")

    if metadata_fields is not None and exclude_metadata_fields is not None:
        raise ValueError(
            "'metadata_fields' and 'exclude_metadata_fields' are mutually exclusive. Set only one."
        )

    dropped = set(reserved_metadata_keys)
    if not include_sys_metadata:
        dropped |= SYS_METADATA_KEYS

    data: dict[str, Any] = {page_content_field: document.page_content}
    metadata = document.metadata or {}

    if metadata_fields is not None:
        selected = {
            field: metadata[field]
            for field in metadata_fields
            if field in metadata and field not in dropped
        }
    else:
        exclude = dropped | set(exclude_metadata_fields or [])
        selected = {field: value for field, value in metadata.items() if field not in exclude}

    if page_content_field in selected:
        raise ValueError(
            f"Metadata key '{page_content_field}' collides with page_content_field. "
            f"Exclude it via 'exclude_metadata_fields' or use a custom "
            f"'document_mapper'."
        )

    data.update(selected)
    return data

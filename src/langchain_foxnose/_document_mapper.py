"""Pure function to map FoxNose search results to LangChain Document objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from langchain_core.documents import Document


def map_results_to_documents(
    results: list[dict[str, Any]],
    *,
    page_content_field: str | None = None,
    page_content_fields: list[str] | None = None,
    page_content_separator: str = "\n\n",
    page_content_mapper: Callable[[dict[str, Any]], str] | None = None,
    metadata_fields: list[str] | None = None,
    exclude_metadata_fields: list[str] | None = None,
    include_sys_metadata: bool = True,
) -> list[Document]:
    """Convert FoxNose search results into LangChain Documents.

    This is a pure function with no side effects.

    Each result is expected to have the FoxNose shape::

        {"_sys": {"key": "...", ...}, "data": {"title": "...", "body": "...", ...}}

    Args:
        results: List of result dicts from the FoxNose ``_search`` response.
        page_content_field: Single ``data`` field to use as ``page_content``.
        page_content_fields: Multiple ``data`` fields concatenated with *separator*.
        page_content_separator: Separator when using *page_content_fields*.
        page_content_mapper: Custom callable ``(result) -> str`` for full control.
        metadata_fields: Whitelist of ``data`` fields to include in metadata.
        exclude_metadata_fields: Blacklist of ``data`` fields to exclude from metadata.
        include_sys_metadata: Whether to include ``_sys`` fields in metadata.

    Returns:
        A list of :class:`langchain_core.documents.Document` objects.
    """
    documents: list[Document] = []

    for result in results:
        page_content = _extract_page_content(
            result,
            page_content_field=page_content_field,
            page_content_fields=page_content_fields,
            page_content_separator=page_content_separator,
            page_content_mapper=page_content_mapper,
        )
        metadata = _extract_metadata(
            result,
            page_content_field=page_content_field,
            page_content_fields=page_content_fields,
            page_content_mapper=page_content_mapper,
            metadata_fields=metadata_fields,
            exclude_metadata_fields=exclude_metadata_fields,
            include_sys_metadata=include_sys_metadata,
        )
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def _extract_page_content(
    result: dict[str, Any],
    *,
    page_content_field: str | None,
    page_content_fields: list[str] | None,
    page_content_separator: str,
    page_content_mapper: Callable[[dict[str, Any]], str] | None,
) -> str:
    """Extract page_content from a single result."""
    if page_content_mapper is not None:
        return page_content_mapper(result)

    data = result.get("data") or {}

    if page_content_field is not None:
        value = data.get(page_content_field, "")
        return value if isinstance(value, str) else str(value) if value is not None else ""

    if page_content_fields is not None:
        parts: list[str] = []
        for field in page_content_fields:
            value = data.get(field)
            if value is not None:
                parts.append(value if isinstance(value, str) else str(value))
        return page_content_separator.join(parts)

    return ""


def _extract_metadata(
    result: dict[str, Any],
    *,
    page_content_field: str | None,
    page_content_fields: list[str] | None,
    page_content_mapper: Callable[[dict[str, Any]], str] | None,
    metadata_fields: list[str] | None,
    exclude_metadata_fields: list[str] | None,
    include_sys_metadata: bool,
) -> dict[str, Any]:
    """Extract metadata from a single result."""
    metadata: dict[str, Any] = {}

    # System metadata
    if include_sys_metadata:
        sys_data = result.get("_sys", {})
        for key in ("key", "folder", "created_at", "updated_at"):
            if key in sys_data:
                metadata[key] = sys_data[key]

    # Data fields for metadata
    data = result.get("data") or {}
    content_fields = _get_content_fields(
        page_content_field=page_content_field,
        page_content_fields=page_content_fields,
        page_content_mapper=page_content_mapper,
    )

    if metadata_fields is not None:
        # Whitelist mode: only include specified fields
        for field in metadata_fields:
            if field in data:
                metadata[field] = data[field]
    else:
        # Include all data fields except content fields and excluded fields
        exclude_set = set(exclude_metadata_fields or [])
        for field, value in data.items():
            if field in content_fields:
                continue
            if field in exclude_set:
                continue
            metadata[field] = value

    return metadata


def _get_content_fields(
    *,
    page_content_field: str | None,
    page_content_fields: list[str] | None,
    page_content_mapper: Callable[[dict[str, Any]], str] | None,
) -> Sequence[str]:
    """Return the set of field names used for page_content (to exclude from metadata)."""
    if page_content_mapper is not None:
        # With a custom mapper, we don't know which fields are used
        return ()
    if page_content_field is not None:
        return (page_content_field,)
    if page_content_fields is not None:
        return page_content_fields
    return ()

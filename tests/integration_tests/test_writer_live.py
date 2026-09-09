"""Live tests for FoxNoseWriter against a configured FoxNose environment.

Every write uses a fresh uuid so reruns never collide. Flux has no delete
endpoint, so these tests APPEND to FOXNOSE_WRITE_COLLECTION_PATH and can never
clean up -- that collection must be a dedicated throwaway, and needs periodic
pruning out of band.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from foxnose_sdk.errors import ContentValidationFailed, ExternalIdConflict
from langchain_core.documents import Document

from langchain_foxnose import FoxNoseBatchWriteError, FoxNoseWriter


def _doc(text: str, **metadata: Any) -> Document:
    return Document(page_content=text, metadata=metadata)


def test_add_documents_creates_readable_resources(
    live_writer: FoxNoseWriter,
    write_flux_client: Any,
    write_collection_path: str,
    write_content_field: str,
) -> None:
    marker = uuid.uuid4().hex
    keys = live_writer.add_documents([_doc(f"first {marker}"), _doc(f"second {marker}")])
    assert len(keys) == 2
    assert len(set(keys)) == 2

    for key, expected in zip(keys, ["first", "second"], strict=True):
        resource = write_flux_client.get_resource(write_collection_path, key)
        data = resource.get("data", resource)
        assert data[write_content_field].startswith(expected)


def test_empty_batch_is_a_no_op(live_writer: FoxNoseWriter) -> None:
    assert live_writer.add_documents([]) == []


def test_external_id_deduplicates_and_stays_out_of_data(
    live_writer: FoxNoseWriter, write_flux_client: Any, write_collection_path: str
) -> None:
    source_id = f"dedup-{uuid.uuid4().hex}"
    [key] = live_writer.add_documents([_doc("deduplicated", source_id=source_id)])

    data = write_flux_client.get_resource(write_collection_path, key).get("data", {})
    assert "source_id" not in data, "the external id must not land in the document"

    with pytest.raises(FoxNoseBatchWriteError) as exc:
        live_writer.add_documents([_doc("second attempt", source_id=source_id)])
    assert isinstance(exc.value.cause, ExternalIdConflict), exc.value.cause
    assert exc.value.written_keys == []


def test_update_replaces_content(
    live_writer: FoxNoseWriter,
    write_flux_client: Any,
    write_collection_path: str,
    write_content_field: str,
) -> None:
    [key] = live_writer.add_documents([_doc(f"before {uuid.uuid4().hex}")])
    after = f"after {uuid.uuid4().hex}"
    assert live_writer.update_document(key, _doc(after))

    data = write_flux_client.get_resource(write_collection_path, key).get("data", {})
    assert data[write_content_field] == after


def test_schema_violation_writes_nothing(
    write_flux_client: Any, write_collection_path: str
) -> None:
    """A field name with a random suffix cannot exist in any schema.

    Asserting the TYPED exception, not just the status code: a Flux write
    reports "data_validation_error", which foxnose-sdk only began mapping onto
    ContentValidationFailed in 0.8.1. This test is what would catch that
    mapping regressing.
    """
    marker = uuid.uuid4().hex
    writer = FoxNoseWriter(
        client=write_flux_client,
        collection_path=write_collection_path,
        document_mapper=lambda _doc: {f"nonexistent_{marker}": "x"},
    )
    with pytest.raises(FoxNoseBatchWriteError) as exc:
        writer.add_documents([_doc("anything")])
    assert exc.value.written_keys == []
    assert exc.value.failed_index == 0
    assert exc.value.pending_indexes == []
    cause = exc.value.cause
    assert isinstance(cause, ContentValidationFailed), (type(cause).__name__, cause)
    assert cause.status_code == 422, cause
    assert cause.errors, "the typed exception carried no structured errors"


def test_partial_batch_reports_all_three_ranges(
    write_flux_client: Any, write_collection_path: str, write_content_field: str
) -> None:
    """[valid, invalid, valid]: the third must never be attempted."""
    marker = uuid.uuid4().hex
    third_id = f"never-{marker}"

    def mapper(document: Document) -> dict[str, Any]:
        if document.page_content == "invalid":
            return {f"nonexistent_{marker}": "x"}
        return {write_content_field: document.page_content}

    writer = FoxNoseWriter(
        client=write_flux_client,
        collection_path=write_collection_path,
        document_mapper=mapper,
        external_id_key="source_id",
    )
    with pytest.raises(FoxNoseBatchWriteError) as exc:
        writer.add_documents(
            [
                _doc(f"valid one {marker}", source_id=f"ok-{marker}"),
                _doc("invalid", source_id=f"bad-{marker}"),
                _doc(f"valid three {marker}", source_id=third_id),
            ]
        )
    assert len(exc.value.written_keys) == 1
    assert exc.value.failed_index == 1
    assert exc.value.pending_indexes == [2]

    # The first really landed and is NOT rolled back...
    written = write_flux_client.get_resource(write_collection_path, exc.value.written_keys[0]).get(
        "data", {}
    )
    assert written[write_content_field] == f"valid one {marker}"

    # ...and the third was never attempted: its external id is still free.
    reuse = FoxNoseWriter(
        client=write_flux_client,
        collection_path=write_collection_path,
        page_content_field=write_content_field,
        external_id_key="source_id",
    )
    assert reuse.add_documents([_doc("proves it was free", source_id=third_id)])


def test_local_mapping_error_writes_nothing(
    write_flux_client: Any, write_collection_path: str, write_content_field: str
) -> None:
    """A bad external id is caught before the first request is sent."""
    writer = FoxNoseWriter(
        client=write_flux_client,
        collection_path=write_collection_path,
        page_content_field=write_content_field,
        external_id_key="source_id",
    )
    marker = uuid.uuid4().hex
    with pytest.raises(TypeError, match="Document 1"):
        writer.add_documents(
            [
                _doc(f"would be first {marker}", source_id=f"a-{marker}"),
                _doc("second", source_id={"not": "scalar"}),
            ]
        )
    # Nothing was written, so the first document's id is still free.
    assert writer.add_documents([_doc("free", source_id=f"a-{marker}")])


async def test_async_add_and_update(
    async_write_flux_client: Any, write_collection_path: str, write_content_field: str
) -> None:
    writer = FoxNoseWriter(
        async_client=async_write_flux_client,
        collection_path=write_collection_path,
        page_content_field=write_content_field,
    )
    marker = uuid.uuid4().hex
    keys = await writer.aadd_documents([_doc(f"async {marker}")])
    assert len(keys) == 1
    assert await writer.aupdate_document(keys[0], _doc(f"async updated {marker}"))


def test_round_trip_loader_to_writer(
    live_writer: FoxNoseWriter,
    write_flux_client: Any,
    write_collection_path: str,
    write_content_field: str,
) -> None:
    """Read a document back and write it out again.

    The regression this guards: the read path injects _sys metadata, none of it
    schema fields -- without the writer stripping it, this fails validation.
    """
    from langchain_foxnose import FoxNoseLoader

    marker = uuid.uuid4().hex
    [key] = live_writer.add_documents([_doc(f"round trip {marker}")])

    loader = FoxNoseLoader(
        client=write_flux_client,
        collection_path=write_collection_path,
        page_content_field=write_content_field,
    )
    loaded = next(doc for doc in loader.load() if doc.metadata.get("key") == key)
    assert "key" in loaded.metadata, "the read path should carry _sys metadata"

    loaded.page_content = f"{loaded.page_content} (edited)"
    assert live_writer.update_document(loaded.metadata["key"], loaded)

    data = write_flux_client.get_resource(write_collection_path, key).get("data", {})
    assert data[write_content_field].endswith("(edited)")

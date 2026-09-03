"""Wire-level checks that FoxNoseWriter's writes hit the right endpoints.

The mock-based tests in ``test_writer.py`` assert on the arguments handed to
``create_resource`` / ``update_resource``.  They would keep passing if the SDK
changed how it addresses a collection or how it envelopes the payload, because
a ``MagicMock`` accepts anything.  These tests drive a real
:class:`~foxnose_sdk.flux.FluxClient` over an ``httpx.MockTransport`` and
assert on the outgoing HTTP method, path and body.
"""

from __future__ import annotations

from typing import Any

from foxnose_sdk.auth import SimpleKeyAuth
from foxnose_sdk.flux import AsyncFluxClient, FluxClient
from langchain_core.documents import Document

from langchain_foxnose import FoxNoseWriter
from tests.conftest import WIRE_BASE_URL, inject_wire_transport, make_wire_recorder

WRITE_RESPONSE = {
    "resource_key": "res_9",
    "revision_key": "rev_9",
    "write_units": 1,
    "published": True,
}


def _sync_writer(transport: Any, **kwargs: Any) -> FoxNoseWriter:
    client = FluxClient(base_url=WIRE_BASE_URL, api_prefix="api", auth=SimpleKeyAuth("pk", "sk"))
    inject_wire_transport(client, transport)
    return FoxNoseWriter(
        client=client, collection_path="articles", page_content_field="body", **kwargs
    )


def _async_writer(transport: Any, **kwargs: Any) -> FoxNoseWriter:
    client = AsyncFluxClient(
        base_url=WIRE_BASE_URL, api_prefix="api", auth=SimpleKeyAuth("pk", "sk")
    )
    inject_wire_transport(client, transport, is_async=True)
    return FoxNoseWriter(
        async_client=client, collection_path="articles", page_content_field="body", **kwargs
    )


class TestCreateOnTheWire:
    def test_posts_to_the_collection_with_a_data_envelope(self) -> None:
        recorded, transport = make_wire_recorder(WRITE_RESPONSE)
        writer = _sync_writer(transport)

        keys = writer.add_documents([Document(page_content="Hello", metadata={"title": "T"})])

        assert keys == ["res_9"]
        assert recorded == [
            {
                "method": "POST",
                "path": "/api/articles/",
                "params": {},
                # The SDK wraps our mapping in {"data": ...}; we must not do it too.
                "body": {"data": {"body": "Hello", "title": "T"}},
            }
        ]

    def test_external_id_goes_in_the_body_as_key_not_into_data(self) -> None:
        recorded, transport = make_wire_recorder(WRITE_RESPONSE)
        writer = _sync_writer(transport, external_id_key="source_id")

        writer.add_documents(
            [Document(page_content="Hello", metadata={"title": "T", "source_id": "s-1"})]
        )

        assert recorded[0]["body"] == {
            "data": {"body": "Hello", "title": "T"},
            "key": "s-1",
        }

    def test_nested_collection_path_is_preserved(self) -> None:
        recorded, transport = make_wire_recorder(WRITE_RESPONSE)
        client = FluxClient(
            base_url=WIRE_BASE_URL, api_prefix="api", auth=SimpleKeyAuth("pk", "sk")
        )
        inject_wire_transport(client, transport)
        writer = FoxNoseWriter(
            client=client, collection_path="users/usr_1/memories", page_content_field="body"
        )

        writer.add_documents([Document(page_content="Hello")])

        assert recorded[0]["path"] == "/api/users/usr_1/memories/"

    def test_batch_issues_one_request_per_document_in_order(self) -> None:
        recorded, transport = make_wire_recorder(WRITE_RESPONSE)
        writer = _sync_writer(transport)

        writer.add_documents(
            [Document(page_content="A"), Document(page_content="B"), Document(page_content="C")]
        )

        assert [r["body"]["data"]["body"] for r in recorded] == ["A", "B", "C"]

    def test_empty_batch_sends_nothing(self) -> None:
        recorded, transport = make_wire_recorder(WRITE_RESPONSE)
        assert _sync_writer(transport).add_documents([]) == []
        assert recorded == []


class TestUpdateOnTheWire:
    def test_puts_to_the_resource_with_a_data_envelope(self) -> None:
        recorded, transport = make_wire_recorder(WRITE_RESPONSE)
        writer = _sync_writer(transport)

        revision = writer.update_document(
            "res_1", Document(page_content="New", metadata={"title": "T2"})
        )

        assert revision == "rev_9"
        assert recorded == [
            {
                "method": "PUT",
                "path": "/api/articles/res_1/",
                "params": {},
                "body": {"data": {"body": "New", "title": "T2"}},
            }
        ]

    def test_update_never_sends_an_external_key(self) -> None:
        """update_resource addresses by internal key; an external one is not a thing."""
        recorded, transport = make_wire_recorder(WRITE_RESPONSE)
        writer = _sync_writer(transport, external_id_key="source_id")

        writer.update_document("res_1", Document(page_content="New", metadata={"source_id": "s-1"}))

        assert recorded[0]["body"] == {"data": {"body": "New"}}


class TestAsyncOnTheWire:
    async def test_create(self) -> None:
        recorded, transport = make_wire_recorder(WRITE_RESPONSE)
        writer = _async_writer(transport, external_id_key="source_id")

        keys = await writer.aadd_documents(
            [Document(page_content="Hello", metadata={"source_id": 42})]
        )

        assert keys == ["res_9"]
        assert recorded[0]["method"] == "POST"
        assert recorded[0]["body"] == {"data": {"body": "Hello"}, "key": "42"}

    async def test_update(self) -> None:
        recorded, transport = make_wire_recorder(WRITE_RESPONSE)
        writer = _async_writer(transport)

        revision = await writer.aupdate_document("res_1", Document(page_content="New"))

        assert revision == "rev_9"
        assert recorded[0]["method"] == "PUT"
        assert recorded[0]["path"] == "/api/articles/res_1/"

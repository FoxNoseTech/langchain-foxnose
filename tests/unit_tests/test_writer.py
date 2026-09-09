"""Tests for FoxNoseWriter."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from langchain_foxnose import FoxNoseBatchWriteError, FoxNoseWriter

DOCS = [
    Document(page_content="First", metadata={"title": "One", "category": "tech"}),
    Document(page_content="Second", metadata={"title": "Two", "category": "guide"}),
]

THREE_DOCS = [
    Document(page_content="First", metadata={"title": "One"}),
    Document(page_content="Second", metadata={"title": "Two"}),
    Document(page_content="Third", metadata={"title": "Three"}),
]


class TestValidation:
    def test_requires_a_client(self) -> None:
        with pytest.raises(ValueError, match="At least one of 'client'"):
            FoxNoseWriter(collection_path="articles", page_content_field="body")

    def test_requires_a_content_strategy(self, mock_write_client: MagicMock) -> None:
        with pytest.raises(ValueError, match="Exactly one content mapping strategy"):
            FoxNoseWriter(client=mock_write_client, collection_path="articles")

    def test_rejects_two_content_strategies(self, mock_write_client: MagicMock) -> None:
        with pytest.raises(ValueError, match="Only one content mapping strategy"):
            FoxNoseWriter(
                client=mock_write_client,
                collection_path="articles",
                page_content_field="body",
                document_mapper=lambda d: {"body": d.page_content},
            )

    def test_rejects_mutually_exclusive_metadata_options(
        self, mock_write_client: MagicMock
    ) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            FoxNoseWriter(
                client=mock_write_client,
                collection_path="articles",
                page_content_field="body",
                metadata_fields=["title"],
                exclude_metadata_fields=["category"],
            )

    def test_has_no_concurrency_knob(self, mock_write_client: MagicMock) -> None:
        """Regression guard: concurrency was removed deliberately."""
        with pytest.raises(TypeError):
            FoxNoseWriter(
                client=mock_write_client,
                collection_path="articles",
                page_content_field="body",
                max_concurrency=4,
            )


class TestAddDocuments:
    def test_returns_resource_keys_in_order(self, mock_write_client: MagicMock) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        assert writer.add_documents(DOCS) == ["res_1", "res_2"]

    def test_sends_mapped_data(self, mock_write_client: MagicMock) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        writer.add_documents(DOCS[:1])
        args, kwargs = mock_write_client.create_resource.call_args
        assert args[0] == "articles"
        assert args[1] == {"body": "First", "title": "One", "category": "tech"}
        assert "key" not in kwargs

    def test_empty_batch_makes_no_calls(self, mock_write_client: MagicMock) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        assert writer.add_documents([]) == []
        mock_write_client.create_resource.assert_not_called()

    def test_external_id_key_becomes_key_and_leaves_data(
        self, mock_write_client: MagicMock
    ) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
            external_id_key="source_id",
        )
        writer.add_documents(
            [Document(page_content="X", metadata={"source_id": "s-1", "title": "T"})]
        )
        args, kwargs = mock_write_client.create_resource.call_args
        assert kwargs["key"] == "s-1"
        assert args[1] == {"body": "X", "title": "T"}

    def test_external_id_key_absent_is_not_an_error(self, mock_write_client: MagicMock) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
            external_id_key="source_id",
        )
        writer.add_documents([Document(page_content="X", metadata={"title": "T"})])
        _args, kwargs = mock_write_client.create_resource.call_args
        assert "key" not in kwargs

    def test_external_id_none_is_treated_as_absent(self, mock_write_client: MagicMock) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
            external_id_key="source_id",
        )
        writer.add_documents([Document(page_content="X", metadata={"source_id": None})])
        _args, kwargs = mock_write_client.create_resource.call_args
        assert "key" not in kwargs

    def test_external_id_int_is_stringified(self, mock_write_client: MagicMock) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
            external_id_key="source_id",
        )
        writer.add_documents([Document(page_content="X", metadata={"source_id": 42})])
        _args, kwargs = mock_write_client.create_resource.call_args
        assert kwargs["key"] == "42"

    @pytest.mark.parametrize("bad", [{"a": 1}, ["x"], 1.5, True, object()])
    def test_external_id_non_scalar_raises_typeerror(
        self, mock_write_client: MagicMock, bad: Any
    ) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
            external_id_key="source_id",
        )
        with pytest.raises(TypeError, match="must be a str or int"):
            writer.add_documents([Document(page_content="X", metadata={"source_id": bad})])
        mock_write_client.create_resource.assert_not_called()

    def test_sys_metadata_stripped_by_default(self, mock_write_client: MagicMock) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        writer.add_documents([Document(page_content="X", metadata={"key": "abc", "title": "T"})])
        args, _kwargs = mock_write_client.create_resource.call_args
        assert args[1] == {"body": "X", "title": "T"}

    def test_requires_sync_client(self, mock_async_write_client: AsyncMock) -> None:
        writer = FoxNoseWriter(
            async_client=mock_async_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        with pytest.raises(ValueError, match="Synchronous writing requires"):
            writer.add_documents(DOCS)


class TestLocalErrorsWriteNothing:
    """A mapping error anywhere in the batch must write zero documents."""

    def test_bad_external_id_on_second_document(self, mock_write_client: MagicMock) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
            external_id_key="source_id",
        )
        with pytest.raises(TypeError, match="Document 1"):
            writer.add_documents(
                [
                    Document(page_content="A", metadata={"source_id": "ok"}),
                    Document(page_content="B", metadata={"source_id": {"bad": 1}}),
                ]
            )
        mock_write_client.create_resource.assert_not_called()

    def test_content_collision_on_second_document(self, mock_write_client: MagicMock) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        with pytest.raises(ValueError, match="collides with page_content_field"):
            writer.add_documents(
                [
                    Document(page_content="A", metadata={"title": "T"}),
                    Document(page_content="B", metadata={"body": "clash"}),
                ]
            )
        mock_write_client.create_resource.assert_not_called()

    def test_raising_mapper_on_second_document(self, mock_write_client: MagicMock) -> None:
        def _mapper(document: Document) -> dict[str, Any]:
            if document.page_content == "B":
                raise ValueError("mapper blew up")
            return {"body": document.page_content}

        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            document_mapper=_mapper,
        )
        with pytest.raises(ValueError, match="mapper blew up"):
            writer.add_documents([Document(page_content="A"), Document(page_content="B")])
        mock_write_client.create_resource.assert_not_called()

    async def test_async_bad_external_id_writes_nothing(
        self, mock_async_write_client: AsyncMock
    ) -> None:
        writer = FoxNoseWriter(
            async_client=mock_async_write_client,
            collection_path="articles",
            page_content_field="body",
            external_id_key="source_id",
        )
        with pytest.raises(TypeError, match="Document 1"):
            await writer.aadd_documents(
                [
                    Document(page_content="A", metadata={"source_id": "ok"}),
                    Document(page_content="B", metadata={"source_id": ["bad"]}),
                ]
            )
        mock_async_write_client.create_resource.assert_not_called()

    def test_malformed_success_response_is_wrapped(self, mock_write_client: MagicMock) -> None:
        """A response without resource_key must not escape as a bare KeyError."""
        mock_write_client.create_resource.side_effect = [
            {"resource_key": "res_1", "revision_key": "rev_1"},
            {"revision_key": "rev_2"},  # no resource_key
        ]
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        with pytest.raises(FoxNoseBatchWriteError) as exc:
            writer.add_documents(DOCS)
        assert exc.value.written_keys == ["res_1"]
        assert exc.value.failed_index == 1
        assert isinstance(exc.value.cause, KeyError)


class TestPartialFailure:
    def test_reports_written_keys_and_index(self, mock_write_client: MagicMock) -> None:
        mock_write_client.create_resource.side_effect = [
            {"resource_key": "res_1", "revision_key": "rev_1"},
            RuntimeError("boom"),
        ]
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        with pytest.raises(FoxNoseBatchWriteError) as exc:
            writer.add_documents(DOCS)
        assert exc.value.written_keys == ["res_1"]
        assert exc.value.failed_index == 1
        assert exc.value.total == 2
        assert exc.value.pending_indexes == []
        assert isinstance(exc.value.__cause__, RuntimeError)
        assert exc.value.cause is exc.value.__cause__

    def test_stops_early_and_reports_pending(self, mock_write_client: MagicMock) -> None:
        """The middle document fails; the third must never be attempted."""
        mock_write_client.create_resource.side_effect = [
            {"resource_key": "res_1", "revision_key": "rev_1"},
            RuntimeError("boom"),
            {"resource_key": "res_3", "revision_key": "rev_3"},
        ]
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        with pytest.raises(FoxNoseBatchWriteError) as exc:
            writer.add_documents(THREE_DOCS)
        assert exc.value.written_keys == ["res_1"]
        assert exc.value.failed_index == 1
        assert exc.value.pending_indexes == [2]
        assert mock_write_client.create_resource.call_count == 2

    def test_single_document_failure_also_wraps(self, mock_write_client: MagicMock) -> None:
        mock_write_client.create_resource.side_effect = RuntimeError("boom")
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        with pytest.raises(FoxNoseBatchWriteError) as exc:
            writer.add_documents(DOCS[:1])
        assert exc.value.written_keys == []
        assert exc.value.failed_index == 0
        assert exc.value.pending_indexes == []

    def test_message_names_all_three_ranges(self, mock_write_client: MagicMock) -> None:
        mock_write_client.create_resource.side_effect = [
            {"resource_key": "res_1", "revision_key": "rev_1"},
            RuntimeError("boom"),
            {"resource_key": "res_3", "revision_key": "rev_3"},
        ]
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        with pytest.raises(FoxNoseBatchWriteError) as exc:
            writer.add_documents(THREE_DOCS)
        message = str(exc.value)
        assert "not rolled back" in message
        assert "unknown outcome" in message
        assert "not attempted" in message


class TestUpdateDocument:
    def test_returns_revision_key(self, mock_write_client: MagicMock) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        assert writer.update_document("res_1", DOCS[0]) == "rev_2"

    def test_sends_mapped_data(self, mock_write_client: MagicMock) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        writer.update_document("res_1", DOCS[0])
        args, _kwargs = mock_write_client.update_resource.call_args
        assert args == (
            "articles",
            "res_1",
            {"body": "First", "title": "One", "category": "tech"},
        )

    def test_ignores_external_id_key(self, mock_write_client: MagicMock) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
            external_id_key="source_id",
        )
        writer.update_document("res_1", Document(page_content="X", metadata={"source_id": "s-1"}))
        args, kwargs = mock_write_client.update_resource.call_args
        assert args[2] == {"body": "X"}
        assert "key" not in kwargs

    def test_requires_sync_client(self, mock_async_write_client: AsyncMock) -> None:
        writer = FoxNoseWriter(
            async_client=mock_async_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        with pytest.raises(ValueError, match="Synchronous writing requires"):
            writer.update_document("res_1", DOCS[0])


class TestAsync:
    async def test_aadd_documents(self, mock_async_write_client: AsyncMock) -> None:
        writer = FoxNoseWriter(
            async_client=mock_async_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        assert await writer.aadd_documents(DOCS) == ["res_1", "res_2"]

    async def test_aadd_documents_empty_makes_no_calls(
        self, mock_async_write_client: AsyncMock
    ) -> None:
        writer = FoxNoseWriter(
            async_client=mock_async_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        assert await writer.aadd_documents([]) == []
        mock_async_write_client.create_resource.assert_not_called()

    async def test_aupdate_document(self, mock_async_write_client: AsyncMock) -> None:
        writer = FoxNoseWriter(
            async_client=mock_async_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        assert await writer.aupdate_document("res_1", DOCS[0]) == "rev_2"

    async def test_requires_async_client(self, mock_write_client: MagicMock) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        with pytest.raises(ValueError, match="Async writing requires"):
            await writer.aadd_documents(DOCS)

    async def test_aupdate_requires_async_client(self, mock_write_client: MagicMock) -> None:
        writer = FoxNoseWriter(
            client=mock_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        with pytest.raises(ValueError, match="Async writing requires"):
            await writer.aupdate_document("res_1", DOCS[0])

    async def test_partial_failure_stops_early(self, mock_async_write_client: AsyncMock) -> None:
        """Async must stop at the first failure, like sync -- no gather()."""
        calls: list[Any] = []

        async def _create(*args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(args)
            if len(calls) == 2:
                raise RuntimeError("boom")
            return {"resource_key": f"res_{len(calls)}", "revision_key": "rev_1"}

        mock_async_write_client.create_resource.side_effect = _create
        writer = FoxNoseWriter(
            async_client=mock_async_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        with pytest.raises(FoxNoseBatchWriteError) as exc:
            await writer.aadd_documents(THREE_DOCS)
        assert exc.value.written_keys == ["res_1"]
        assert exc.value.failed_index == 1
        assert exc.value.pending_indexes == [2]
        assert len(calls) == 2, "the third write must never have been attempted"

    async def test_slow_first_write_does_not_reorder(
        self, mock_async_write_client: AsyncMock
    ) -> None:
        """A slow first write must not let the second overtake it."""
        order: list[int] = []

        async def _create(*args: Any, **kwargs: Any) -> dict[str, Any]:
            index = len(order)
            if index == 0:
                await asyncio.sleep(0.05)
            order.append(index)
            return {"resource_key": f"res_{index}", "revision_key": "rev"}

        mock_async_write_client.create_resource.side_effect = _create
        writer = FoxNoseWriter(
            async_client=mock_async_write_client,
            collection_path="articles",
            page_content_field="body",
        )
        assert await writer.aadd_documents(DOCS) == ["res_0", "res_1"]
        assert order == [0, 1]


class TestFromClientParams:
    def test_builds_sync_client(self) -> None:
        from foxnose_sdk.auth import SimpleKeyAuth

        writer = FoxNoseWriter.from_client_params(
            base_url="https://example.com",
            api_prefix="my_api",
            auth=SimpleKeyAuth("pk", "sk"),
            collection_path="articles",
            page_content_field="body",
        )
        assert writer.client is not None
        assert writer.async_client is None
        assert writer.collection_path == "articles"

    def test_builds_async_client(self) -> None:
        from foxnose_sdk.auth import SimpleKeyAuth

        writer = FoxNoseWriter.from_client_params(
            base_url="https://example.com",
            api_prefix="my_api",
            auth=SimpleKeyAuth("pk", "sk"),
            collection_path="articles",
            page_content_field="body",
            async_mode=True,
        )
        assert writer.async_client is not None
        assert writer.client is None


class TestExternalIdKeyValidation:
    def test_empty_external_id_key_is_rejected(self, mock_flux_client: Any) -> None:
        """Neither None nor usable: it used to be read two different ways."""
        with pytest.raises(ValueError, match="non-empty metadata key"):
            FoxNoseWriter(
                client=mock_flux_client,
                collection_path="kb",
                page_content_field="body",
                external_id_key="",
            )

    def test_whitespace_external_id_key_is_rejected(self, mock_flux_client: Any) -> None:
        with pytest.raises(ValueError, match="non-empty metadata key"):
            FoxNoseWriter(
                client=mock_flux_client,
                collection_path="kb",
                page_content_field="body",
                external_id_key="   ",
            )

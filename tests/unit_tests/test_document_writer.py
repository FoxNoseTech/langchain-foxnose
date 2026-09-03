"""Tests for the pure Document -> FoxNose data mapper."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from langchain_foxnose._document_writer import SYS_METADATA_KEYS, map_document_to_data


def _doc(**metadata: Any) -> Document:
    return Document(page_content="Hello world", metadata=metadata)


class TestBasicMapping:
    def test_page_content_goes_to_field(self) -> None:
        data = map_document_to_data(_doc(), page_content_field="body")
        assert data == {"body": "Hello world"}

    def test_metadata_is_included(self) -> None:
        data = map_document_to_data(_doc(title="T", category="tech"), page_content_field="body")
        assert data == {"body": "Hello world", "title": "T", "category": "tech"}

    def test_requires_page_content_field(self) -> None:
        with pytest.raises(ValueError, match="page_content_field"):
            map_document_to_data(_doc())

    def test_empty_page_content_is_preserved(self) -> None:
        data = map_document_to_data(Document(page_content=""), page_content_field="body")
        assert data == {"body": ""}


class TestSysMetadata:
    def test_sys_keys_dropped_by_default(self) -> None:
        data = map_document_to_data(
            _doc(key="abc", folder="articles", created_at="x", updated_at="y", title="T"),
            page_content_field="body",
        )
        assert data == {"body": "Hello world", "title": "T"}

    def test_sys_keys_kept_when_opted_in(self) -> None:
        data = map_document_to_data(
            _doc(key="abc", title="T"),
            page_content_field="body",
            include_sys_metadata=True,
        )
        assert data == {"body": "Hello world", "key": "abc", "title": "T"}

    def test_sys_keys_constant(self) -> None:
        assert set(SYS_METADATA_KEYS) == {"key", "folder", "created_at", "updated_at"}


class TestFieldSelection:
    def test_whitelist(self) -> None:
        data = map_document_to_data(
            _doc(title="T", category="tech", status="draft"),
            page_content_field="body",
            metadata_fields=["title", "category"],
        )
        assert data == {"body": "Hello world", "title": "T", "category": "tech"}

    def test_whitelist_skips_absent(self) -> None:
        data = map_document_to_data(
            _doc(title="T"),
            page_content_field="body",
            metadata_fields=["title", "nope"],
        )
        assert data == {"body": "Hello world", "title": "T"}

    def test_blacklist(self) -> None:
        data = map_document_to_data(
            _doc(title="T", category="tech"),
            page_content_field="body",
            exclude_metadata_fields=["category"],
        )
        assert data == {"body": "Hello world", "title": "T"}

    def test_whitelist_and_blacklist_are_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            map_document_to_data(
                _doc(),
                page_content_field="body",
                metadata_fields=["title"],
                exclude_metadata_fields=["category"],
            )

    def test_reserved_keys_dropped_in_blacklist_mode(self) -> None:
        data = map_document_to_data(
            _doc(title="T", ext_id="e1"),
            page_content_field="body",
            reserved_metadata_keys=("ext_id",),
        )
        assert data == {"body": "Hello world", "title": "T"}

    def test_reserved_keys_dropped_in_whitelist_mode(self) -> None:
        data = map_document_to_data(
            _doc(title="T", ext_id="e1"),
            page_content_field="body",
            metadata_fields=["title", "ext_id"],
            reserved_metadata_keys=("ext_id",),
        )
        assert data == {"body": "Hello world", "title": "T"}

    def test_reserved_keys_win_over_sys_opt_in(self) -> None:
        data = map_document_to_data(
            _doc(key="abc", title="T"),
            page_content_field="body",
            include_sys_metadata=True,
            reserved_metadata_keys=("key",),
        )
        assert data == {"body": "Hello world", "title": "T"}

    def test_empty_whitelist_writes_content_only(self) -> None:
        data = map_document_to_data(_doc(title="T"), page_content_field="body", metadata_fields=[])
        assert data == {"body": "Hello world"}


class TestCollisions:
    def test_metadata_key_colliding_with_content_field_raises(self) -> None:
        with pytest.raises(ValueError, match="collides with page_content_field"):
            map_document_to_data(_doc(body="other"), page_content_field="body")

    def test_collision_avoided_by_exclusion(self) -> None:
        data = map_document_to_data(
            _doc(body="other", title="T"),
            page_content_field="body",
            exclude_metadata_fields=["body"],
        )
        assert data == {"body": "Hello world", "title": "T"}

    def test_collision_in_whitelist_mode_also_raises(self) -> None:
        with pytest.raises(ValueError, match="collides with page_content_field"):
            map_document_to_data(
                _doc(body="other"),
                page_content_field="body",
                metadata_fields=["body"],
            )


class TestCustomMapper:
    def test_mapper_wins(self) -> None:
        data = map_document_to_data(
            _doc(title="T"),
            page_content_field="body",
            metadata_fields=["title"],
            document_mapper=lambda d: {"custom": d.page_content.upper()},
        )
        assert data == {"custom": "HELLO WORLD"}

    def test_mapper_result_is_copied(self) -> None:
        shared: dict[str, Any] = {"a": 1}
        data = map_document_to_data(_doc(), document_mapper=lambda d: shared)
        data["b"] = 2
        assert shared == {"a": 1}

    def test_mapper_needs_no_page_content_field(self) -> None:
        data = map_document_to_data(_doc(), document_mapper=lambda d: {"x": d.page_content})
        assert data == {"x": "Hello world"}


class TestRoundTrip:
    def test_loaded_document_round_trips(self) -> None:
        from langchain_foxnose._document_mapper import map_results_to_documents

        result = {
            "_sys": {
                "key": "abc123",
                "folder": "articles",
                "created_at": "2024-06-01T10:00:00Z",
                "updated_at": "2024-06-15T12:00:00Z",
            },
            "data": {"title": "T", "body": "B", "category": "tech"},
        }
        [doc] = map_results_to_documents([result], page_content_field="body")
        # The read path injects the _sys keys by default; they are not schema
        # fields, so a write must not send them back.
        assert "key" in doc.metadata
        assert map_document_to_data(doc, page_content_field="body") == {
            "body": "B",
            "title": "T",
            "category": "tech",
        }

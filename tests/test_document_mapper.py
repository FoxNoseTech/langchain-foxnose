"""Tests for the document mapper."""

from __future__ import annotations

from typing import Any

from langchain_foxnose._document_mapper import map_results_to_documents


class TestSingleFieldContent:
    """page_content_field (single field)."""

    def test_basic_mapping(self, sample_results: list[dict]) -> None:
        docs = map_results_to_documents(sample_results, page_content_field="body")
        assert len(docs) == 3
        assert docs[0].page_content == "FoxNose is a serverless knowledge platform..."
        assert docs[1].page_content == "Learn how to use vector search in FoxNose..."

    def test_missing_field_returns_empty_string(self) -> None:
        results = [{"_sys": {"key": "a"}, "data": {"title": "Hello"}}]
        docs = map_results_to_documents(results, page_content_field="body")
        assert docs[0].page_content == ""

    def test_non_string_field_converted(self) -> None:
        results = [{"_sys": {"key": "a"}, "data": {"count": 42}}]
        docs = map_results_to_documents(results, page_content_field="count")
        assert docs[0].page_content == "42"

    def test_none_field_returns_empty_string(self) -> None:
        results = [{"_sys": {"key": "a"}, "data": {"body": None}}]
        docs = map_results_to_documents(results, page_content_field="body")
        assert docs[0].page_content == ""


class TestMultiFieldContent:
    """page_content_fields (multiple fields concatenated)."""

    def test_two_fields_concatenated(self, sample_results: list[dict]) -> None:
        docs = map_results_to_documents(sample_results, page_content_fields=["title", "body"])
        expected = "Getting Started with FoxNose\n\nFoxNose is a serverless knowledge platform..."
        assert docs[0].page_content == expected

    def test_custom_separator(self, sample_results: list[dict]) -> None:
        docs = map_results_to_documents(
            sample_results,
            page_content_fields=["title", "body"],
            page_content_separator=" | ",
        )
        assert " | " in docs[0].page_content

    def test_missing_field_skipped(self) -> None:
        results = [{"_sys": {"key": "a"}, "data": {"title": "Hello"}}]
        docs = map_results_to_documents(results, page_content_fields=["title", "missing"])
        assert docs[0].page_content == "Hello"

    def test_all_fields_missing(self) -> None:
        results = [{"_sys": {"key": "a"}, "data": {}}]
        docs = map_results_to_documents(results, page_content_fields=["title", "body"])
        assert docs[0].page_content == ""


class TestCustomMapper:
    """page_content_mapper (callable)."""

    def test_custom_mapper(self, sample_results: list[dict]) -> None:
        def mapper(result: dict[str, Any]) -> str:
            return f"# {result['data']['title']}\n{result['data']['body']}"

        docs = map_results_to_documents(sample_results, page_content_mapper=mapper)
        assert docs[0].page_content.startswith("# Getting Started with FoxNose")

    def test_custom_mapper_with_sys(self, sample_results: list[dict]) -> None:
        def mapper(result: dict[str, Any]) -> str:
            return f"[{result['_sys']['key']}] {result['data']['title']}"

        docs = map_results_to_documents(sample_results, page_content_mapper=mapper)
        assert docs[0].page_content == "[abc123] Getting Started with FoxNose"


class TestMetadata:
    """Metadata extraction."""

    def test_sys_metadata_included_by_default(self, sample_results: list[dict]) -> None:
        docs = map_results_to_documents(sample_results, page_content_field="body")
        meta = docs[0].metadata
        assert meta["key"] == "abc123"
        assert meta["folder"] == "articles"
        assert meta["created_at"] == "2024-06-01T10:00:00Z"
        assert meta["updated_at"] == "2024-06-15T12:00:00Z"

    def test_sys_metadata_excluded(self, sample_results: list[dict]) -> None:
        docs = map_results_to_documents(
            sample_results, page_content_field="body", include_sys_metadata=False
        )
        meta = docs[0].metadata
        assert "key" not in meta
        assert "folder" not in meta

    def test_data_fields_in_metadata_exclude_content(self, sample_results: list[dict]) -> None:
        docs = map_results_to_documents(sample_results, page_content_field="body")
        meta = docs[0].metadata
        # "body" is page_content, should NOT be in metadata
        assert "body" not in meta
        # Other data fields should be in metadata
        assert meta["title"] == "Getting Started with FoxNose"
        assert meta["category"] == "tech"
        assert meta["status"] == "published"

    def test_metadata_whitelist(self, sample_results: list[dict]) -> None:
        docs = map_results_to_documents(
            sample_results,
            page_content_field="body",
            metadata_fields=["title"],
        )
        meta = docs[0].metadata
        assert "title" in meta
        assert "category" not in meta
        assert "status" not in meta
        # sys metadata still included
        assert "key" in meta

    def test_metadata_blacklist(self, sample_results: list[dict]) -> None:
        docs = map_results_to_documents(
            sample_results,
            page_content_field="body",
            exclude_metadata_fields=["status"],
        )
        meta = docs[0].metadata
        assert "title" in meta
        assert "category" in meta
        assert "status" not in meta

    def test_custom_mapper_includes_all_data_in_metadata(self, sample_results: list[dict]) -> None:
        """When using a custom mapper, all data fields go into metadata
        since we don't know which fields the mapper uses."""
        docs = map_results_to_documents(
            sample_results,
            page_content_mapper=lambda r: r["data"]["body"],
        )
        meta = docs[0].metadata
        # All data fields present because mapper doesn't declare content fields
        assert "body" in meta
        assert "title" in meta
        assert "category" in meta

    def test_multi_field_content_excluded_from_metadata(self, sample_results: list[dict]) -> None:
        docs = map_results_to_documents(sample_results, page_content_fields=["title", "body"])
        meta = docs[0].metadata
        assert "title" not in meta
        assert "body" not in meta
        assert "category" in meta


class TestEmptyResults:
    """Edge cases."""

    def test_empty_results(self) -> None:
        docs = map_results_to_documents([], page_content_field="body")
        assert docs == []

    def test_result_with_empty_data(self) -> None:
        results = [{"_sys": {"key": "a"}, "data": {}}]
        docs = map_results_to_documents(results, page_content_field="body")
        assert len(docs) == 1
        assert docs[0].page_content == ""

    def test_result_with_no_sys(self) -> None:
        results = [{"data": {"body": "content"}}]
        docs = map_results_to_documents(results, page_content_field="body")
        assert docs[0].page_content == "content"
        assert docs[0].metadata == {}

    def test_result_with_none_data(self) -> None:
        """data=None should not raise; treated as empty dict."""
        results = [{"_sys": {"key": "a"}, "data": None}]
        docs = map_results_to_documents(results, page_content_field="body")
        assert len(docs) == 1
        assert docs[0].page_content == ""
        assert docs[0].metadata == {"key": "a"}

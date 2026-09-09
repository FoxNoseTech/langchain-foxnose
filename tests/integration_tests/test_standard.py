"""LangChain's standard retriever suite, run against a configured environment.

The suite asserts EXACTLY 3 and EXACTLY 1 results, which is why the read
collection must contain at least three documents matching FOXNOSE_QUERY and why
`read_corpus_is_sufficient` checks that before anything runs.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_foxnose import FoxNoseRetriever


class TestFoxNoseRetrieverStandard(RetrieversIntegrationTests):
    """The suite reads its configuration off ``self``, so bind the fixtures."""

    @pytest.fixture(autouse=True)
    def _bind(
        self,
        flux_client: Any,
        collection_path: str,
        content_field: str,
        read_corpus_is_sufficient: None,
    ) -> None:
        self._client = flux_client
        self._collection_path = collection_path
        self._content_field = content_field

    @property
    def retriever_constructor(self) -> type[FoxNoseRetriever]:
        return FoxNoseRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return {
            "client": self._client,
            "collection_path": self._collection_path,
            "page_content_field": self._content_field,
            "search_mode": "text",
        }

    @property
    def retriever_query_example(self) -> str:
        return os.environ["FOXNOSE_QUERY"]

    @property
    def num_results_arg_name(self) -> str:
        return "top_k"

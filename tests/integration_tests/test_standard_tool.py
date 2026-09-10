"""LangChain's standard tool suite, invoking half.

The suite calls the tool for real, so it needs the live retriever behind it.
The offline half -- name, schema, construction -- is in
tests/unit_tests/test_standard_tool.py and runs without credentials.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_foxnose import create_foxnose_tool


class TestFoxNoseToolStandard(ToolsIntegrationTests):
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
    def tool_constructor(self) -> BaseTool:
        return create_foxnose_tool(
            client=self._client,
            collection_path=self._collection_path,
            page_content_field=self._content_field,
            search_mode="text",
        )

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": os.environ["FOXNOSE_QUERY"]}

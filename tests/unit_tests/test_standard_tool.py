"""LangChain's standard tool suite, offline half.

These check only shape -- name, input schema, construction -- and never
invoke, so a mock client is enough and the suite runs with sockets disabled.
The invoking half lives in tests/integration_tests/test_standard_tool.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_foxnose import create_foxnose_tool


class TestFoxNoseToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        """An instance, not a class: create_foxnose_tool is a factory.

        The suite requires tool_constructor_params to stay empty in that case,
        which is why the configuration is applied here.
        """
        return create_foxnose_tool(
            client=MagicMock(),
            collection_path="articles",
            page_content_field="body",
        )

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "foxnose"}

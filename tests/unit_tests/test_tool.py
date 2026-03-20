"""Tests for create_foxnose_tool."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.tools import BaseTool

from langchain_foxnose import FoxNoseRetriever, create_foxnose_tool


class TestToolCreation:
    """Tool creation from various inputs."""

    def test_create_from_retriever(self, mock_flux_client: MagicMock) -> None:
        retriever = FoxNoseRetriever(
            client=mock_flux_client,
            folder_path="kb",
            page_content_field="body",
        )
        tool = create_foxnose_tool(retriever=retriever)
        assert isinstance(tool, BaseTool)

    def test_create_from_client_params(self, mock_flux_client: MagicMock) -> None:
        tool = create_foxnose_tool(
            client=mock_flux_client,
            folder_path="kb",
            page_content_field="body",
        )
        assert isinstance(tool, BaseTool)

    def test_raises_without_retriever_or_client(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)either provide"):
            create_foxnose_tool(folder_path="kb", page_content_field="body")

    def test_custom_name(self, mock_flux_client: MagicMock) -> None:
        tool = create_foxnose_tool(
            client=mock_flux_client,
            folder_path="kb",
            page_content_field="body",
            name="my_search",
        )
        assert tool.name == "my_search"

    def test_custom_description(self, mock_flux_client: MagicMock) -> None:
        tool = create_foxnose_tool(
            client=mock_flux_client,
            folder_path="kb",
            page_content_field="body",
            description="Search my docs.",
        )
        assert tool.description == "Search my docs."

    def test_default_name_and_description(self, mock_flux_client: MagicMock) -> None:
        tool = create_foxnose_tool(
            client=mock_flux_client,
            folder_path="kb",
            page_content_field="body",
        )
        assert tool.name == "foxnose_search"
        assert "FoxNose" in tool.description


class TestToolInvocation:
    """Tool invocation."""

    def test_invoke_returns_string(self, mock_flux_client: MagicMock) -> None:
        tool = create_foxnose_tool(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
        )
        result = tool.invoke("test query")
        assert isinstance(result, str)
        assert "FoxNose is a serverless knowledge platform..." in result

    def test_invoke_with_retriever_kwargs(self, mock_flux_client: MagicMock) -> None:
        tool = create_foxnose_tool(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
            search_mode="text",
            top_k=3,
        )
        result = tool.invoke("test query")
        assert isinstance(result, str)

    def test_invoke_empty_results(self, mock_flux_client: MagicMock) -> None:
        mock_flux_client.search.return_value = {"results": [], "limit": 5}
        tool = create_foxnose_tool(
            client=mock_flux_client,
            folder_path="articles",
            page_content_field="body",
        )
        result = tool.invoke("test query")
        assert isinstance(result, str)

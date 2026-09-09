"""Live tests for create_foxnose_tool against a configured FoxNose environment."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from langchain_foxnose import create_foxnose_tool


def test_tool_invoke_returns_a_string(
    flux_client: Any,
    collection_path: str,
    content_field: str,
    query: str,
    read_corpus_is_sufficient: None,
) -> None:
    tool = create_foxnose_tool(
        client=flux_client,
        collection_path=collection_path,
        page_content_field=content_field,
        search_mode="text",
    )
    result = tool.invoke({"query": query})
    assert isinstance(result, str)
    assert result.strip()


def test_tool_from_an_existing_retriever(live_retriever: Any, query: str) -> None:
    tool = create_foxnose_tool(
        retriever=live_retriever,
        name="kb_search",
        description="Search the knowledge base.",
    )
    assert tool.name == "kb_search"
    assert tool.invoke({"query": query}).strip()


def test_content_and_artifact_carries_documents(
    flux_client: Any,
    collection_path: str,
    content_field: str,
    query: str,
    read_corpus_is_sufficient: None,
) -> None:
    """The artifact only materialises when the tool is called as a TOOL CALL.

    A plain tool.invoke({"query": ...}) returns just the joined string, because
    there is no tool call for the artifact to attach to.
    """
    tool = create_foxnose_tool(
        client=flux_client,
        collection_path=collection_path,
        page_content_field=content_field,
        search_mode="text",
        response_format="content_and_artifact",
    )
    assert isinstance(tool.invoke({"query": query}), str)

    message = tool.invoke(
        {"name": tool.name, "args": {"query": query}, "id": "call_1", "type": "tool_call"}
    )
    assert message.content.strip()
    assert message.artifact
    assert all(isinstance(doc, Document) for doc in message.artifact)


async def test_tool_ainvoke(
    async_flux_client: Any,
    collection_path: str,
    content_field: str,
    query: str,
    read_corpus_is_sufficient: None,
) -> None:
    tool = create_foxnose_tool(
        async_client=async_flux_client,
        collection_path=collection_path,
        page_content_field=content_field,
        search_mode="text",
    )
    result = await tool.ainvoke({"query": query})
    assert isinstance(result, str)
    assert result.strip()

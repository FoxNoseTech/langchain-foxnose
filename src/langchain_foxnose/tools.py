"""FoxNose search tool for LangChain agents."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool, create_retriever_tool

from langchain_foxnose.retrievers import FoxNoseRetriever


def create_foxnose_tool(
    *,
    client: Any | None = None,
    async_client: Any | None = None,
    retriever: FoxNoseRetriever | None = None,
    folder_path: str | None = None,
    page_content_field: str | None = None,
    name: str = "foxnose_search",
    description: str = (
        "Search the FoxNose knowledge base. "
        "Use this tool to find relevant information for answering questions."
    ),
    document_separator: str = "\n\n",
    response_format: Literal["content", "content_and_artifact"] = "content",
    **retriever_kwargs: Any,
) -> BaseTool:
    """Create a LangChain tool for FoxNose search.

    Either pass an existing ``retriever`` or provide ``client`` + config
    to create one internally.

    Example:
        .. code-block:: python

            from foxnose_sdk.flux import FluxClient
            from foxnose_sdk.auth import SimpleKeyAuth
            from langchain_foxnose import create_foxnose_tool

            client = FluxClient(
                base_url="https://<env_key>.fxns.io",
                api_prefix="my_api",
                auth=SimpleKeyAuth("pk", "sk"),
            )
            tool = create_foxnose_tool(
                client=client,
                folder_path="knowledge-base",
                page_content_field="body",
            )

    Args:
        client: Synchronous :class:`~foxnose_sdk.flux.FluxClient` instance.
        async_client: Asynchronous :class:`~foxnose_sdk.flux.AsyncFluxClient` instance.
        retriever: An existing :class:`FoxNoseRetriever` to wrap. If provided,
            ``client``, ``async_client``, ``folder_path``, ``page_content_field``,
            and ``retriever_kwargs`` are ignored.
        folder_path: Folder path in FoxNose (required when building a new retriever).
        page_content_field: Single data field for ``page_content``
            (required when building a new retriever, unless another content
            strategy is provided via ``retriever_kwargs``).
        name: Tool name exposed to the LLM agent.
        description: Tool description exposed to the LLM agent.
        document_separator: Separator between documents in the response string.
        response_format: ``"content"`` returns a plain string;
            ``"content_and_artifact"`` returns ``(str, list[Document])``.
        **retriever_kwargs: Extra keyword arguments forwarded to
            :class:`FoxNoseRetriever` when creating a new instance.

    Returns:
        A :class:`~langchain_core.tools.BaseTool` that performs FoxNose search.

    Raises:
        ValueError: If neither ``retriever`` nor ``client``/``async_client``
            is provided.
    """
    if retriever is None:
        if client is None and async_client is None:
            raise ValueError(
                "Either provide a 'retriever' or at least one of "
                "'client' / 'async_client' to build one."
            )
        retriever = FoxNoseRetriever(
            client=client,
            async_client=async_client,
            folder_path=folder_path,  # type: ignore[arg-type]
            page_content_field=page_content_field,
            **retriever_kwargs,
        )

    return create_retriever_tool(
        retriever=retriever,
        name=name,
        description=description,
        document_separator=document_separator,
        response_format=response_format,
    )

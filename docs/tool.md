# Search Tool

`create_foxnose_tool` creates a LangChain tool that wraps `FoxNoseRetriever`, making FoxNose search available to LLM agents.

Under the hood it uses LangChain's built-in `create_retriever_tool`, so the resulting tool is fully compatible with any LangChain agent framework.

## Quick Start

```python
from foxnose_sdk.flux import FluxClient
from foxnose_sdk.auth import SimpleKeyAuth
from langchain_foxnose import create_foxnose_tool

client = FluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)

tool = create_foxnose_tool(
    client=client,
    folder_path="knowledge-base",
    page_content_field="body",
)
```

## Using with an Agent

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
agent = create_agent(model=llm, tools=[tool])

result = agent.invoke({"messages": [{"role": "user", "content": "How do I reset my password?"}]})
print(result["messages"][-1].content)
```

`create_agent` accepts a model instance or a provider string, so
`create_agent(model="openai:gpt-4o", tools=[tool])` works too and saves the
explicit `ChatOpenAI` import.

!!! note "Migrating from LangChain 0.3"

    `langgraph.prebuilt.create_react_agent` is deprecated in favour of
    `langchain.agents.create_agent`, which is now the only public agent factory
    (`langchain.agents.__all__` is just `AgentState` and `create_agent`). Note
    that `create_react_agent` took the model positionally as its first
    argument and called the instruction parameter `prompt`; `create_agent`
    names them `model` and `system_prompt`. The old function still imports and
    emits no runtime warning, so nothing breaks loudly — it just rots.

## Wrapping an Existing Retriever

If you already have a configured `FoxNoseRetriever`, pass it directly:

```python
from langchain_foxnose import FoxNoseRetriever, create_foxnose_tool

retriever = FoxNoseRetriever(
    client=client,
    folder_path="knowledge-base",
    page_content_field="body",
    search_mode="hybrid",
    top_k=5,
)

tool = create_foxnose_tool(
    retriever=retriever,
    name="kb_search",
    description="Search the internal knowledge base for answers.",
)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `client` | `None` | Sync `FluxClient` (used to build a retriever if none provided) |
| `async_client` | `None` | Async `AsyncFluxClient` |
| `retriever` | `None` | Existing `FoxNoseRetriever` to wrap |
| `folder_path` | — | Folder path (required when building a new retriever) |
| `page_content_field` | `None` | Data field for `page_content` |
| `name` | `"foxnose_search"` | Tool name exposed to the agent |
| `description` | *"Search the FoxNose…"* | Tool description exposed to the agent |
| `document_separator` | `"\n\n"` | Separator between documents in the response |
| `response_format` | `"content"` | `"content"` for plain text, `"content_and_artifact"` for `(str, list[Document])` |
| `**retriever_kwargs` | — | Extra args forwarded to `FoxNoseRetriever` |

## Response Formats

- **`"content"`** (default) — The tool returns a plain string with document contents joined by `document_separator`. Best for simple agents.
- **`"content_and_artifact"`** — The tool returns a tuple `(str, list[Document])`. Use this when your agent needs access to raw Document objects and their metadata.

## API Reference

::: langchain_foxnose.tools.create_foxnose_tool

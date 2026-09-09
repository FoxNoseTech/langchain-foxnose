# Examples

## Basic Retrieval

```python
from foxnose_sdk.flux import FluxClient
from foxnose_sdk.auth import SimpleKeyAuth
from langchain_foxnose import FoxNoseRetriever

client = FluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("pk", "sk"),
)

retriever = FoxNoseRetriever(
    client=client,
    collection_path="knowledge-base",
    page_content_field="body",
    search_mode="hybrid",
    top_k=5,
)

docs = retriever.invoke("What is FoxNose?")
for doc in docs:
    print(f"[{doc.metadata['key']}] {doc.page_content[:80]}...")
```

## Hybrid Search with Custom Weights

```python
retriever = FoxNoseRetriever(
    client=client,
    collection_path="articles",
    page_content_field="body",
    search_mode="hybrid",
    top_k=10,
    hybrid_config={
        "vector_weight": 0.7,
        "text_weight": 0.3,
        "rerank_results": True,
    },
)
```

## Filtered Retrieval

```python
retriever = FoxNoseRetriever(
    client=client,
    collection_path="articles",
    page_content_field="body",
    where={
        "$": {
            "all_of": [
                {"status__eq": "published"},
                {"category__in": ["tech", "science"]},
                {"published_at__gte": "2024-01-01"},
            ]
        }
    },
)
```

## Vector-Boosted Search

```python
retriever = FoxNoseRetriever(
    client=client,
    collection_path="articles",
    page_content_field="body",
    search_mode="vector_boosted",
    vector_boost_config={
        "boost_factor": 1.3,
        "similarity_threshold": 0.75,
        "max_boost_results": 15,
    },
)
```

## Async Retrieval

```python
import asyncio
from foxnose_sdk.flux import AsyncFluxClient
from foxnose_sdk.auth import SimpleKeyAuth
from langchain_foxnose import FoxNoseRetriever

async def main():
    async_client = AsyncFluxClient(
        base_url="https://<env_key>.fxns.io",
        api_prefix="my_api",
        auth=SimpleKeyAuth("pk", "sk"),
    )

    retriever = FoxNoseRetriever(
        async_client=async_client,
        collection_path="knowledge-base",
        page_content_field="body",
        search_mode="hybrid",
        top_k=5,
    )

    docs = await retriever.ainvoke("async search query")
    for doc in docs:
        print(doc.page_content[:80])

    await async_client.aclose()

asyncio.run(main())
```

## Multi-Field Content

```python
retriever = FoxNoseRetriever(
    client=client,
    collection_path="articles",
    page_content_fields=["title", "summary", "body"],
    page_content_separator="\n\n",
)
```

## Custom Content Mapper

```python
retriever = FoxNoseRetriever(
    client=client,
    collection_path="articles",
    page_content_mapper=lambda r: (
        f"# {r['data']['title']}\n"
        f"Category: {r['data'].get('category', 'N/A')}\n\n"
        f"{r['data']['body']}"
    ),
)
```

## Agentic RAG

```python
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage

from langchain_foxnose import create_foxnose_tool

tool = create_foxnose_tool(
    client=client,
    collection_path="articles",
    page_content_field="body",
    response_format="content_and_artifact",
)
agent = create_agent(model="openai:gpt-4o", tools=[tool])

result = agent.invoke(
    {"messages": [{"role": "user", "content": "How does vector search work?"}]}
)
print(result["messages"][-1].content)

# Source documents come back on the tool's artifact.
for message in result["messages"]:
    if isinstance(message, ToolMessage) and message.artifact:
        for doc in message.artifact:
            print(f"  Source: {doc.metadata['key']}")
```

`response_format="content_and_artifact"` is what makes the retrieved
`Document` objects available; with the default `"content"` the tool returns
only the joined text. The artifact is attached to the `ToolMessage` the agent
produces — calling `tool.invoke({"query": ...})` yourself returns just the
string, because there is no tool call to attach it to.

!!! note "Migrating from LangChain 0.3"

    This section used `RetrievalQA` from `langchain.chains`, which was removed
    in LangChain 1.0 along with the whole module. `result["source_documents"]`
    has no direct equivalent — the artifact above replaces it.

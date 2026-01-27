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
    folder_path="knowledge-base",
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
    folder_path="articles",
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
    folder_path="articles",
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
    folder_path="articles",
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
        folder_path="knowledge-base",
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
    folder_path="articles",
    page_content_fields=["title", "summary", "body"],
    page_content_separator="\n\n",
)
```

## Custom Content Mapper

```python
retriever = FoxNoseRetriever(
    client=client,
    folder_path="articles",
    page_content_mapper=lambda r: (
        f"# {r['data']['title']}\n"
        f"Category: {r['data'].get('category', 'N/A')}\n\n"
        f"{r['data']['body']}"
    ),
)
```

## RetrievalQA Chain

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=retriever,
    return_source_documents=True,
)

result = qa.invoke({"query": "How does vector search work?"})
print(result["result"])
for doc in result["source_documents"]:
    print(f"  Source: {doc.metadata['key']}")
```

# langchain-foxnose

[![PyPI version](https://img.shields.io/pypi/v/langchain-foxnose.svg)](https://pypi.org/project/langchain-foxnose/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-foxnose.svg)](https://pypi.org/project/langchain-foxnose/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/FoxNoseTech/langchain-foxnose/actions/workflows/ci.yml/badge.svg)](https://github.com/FoxNoseTech/langchain-foxnose/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/FoxNoseTech/langchain-foxnose/branch/main/graph/badge.svg)](https://codecov.io/gh/FoxNoseTech/langchain-foxnose)
[![Docs](https://img.shields.io/badge/docs-langchain--foxnose.readthedocs.io-blue)](https://langchain-foxnose.readthedocs.io/)

LangChain integration for [FoxNose](https://foxnose.net?utm_source=github&utm_medium=repository&utm_campaign=langchain-foxnose) — the serverless knowledge platform purpose-built as the knowledge layer for RAG and AI agents.

- **`FoxNoseRetriever`** — query-based retrieval for RAG pipelines
- **`FoxNoseLoader`** — bulk document loading with cursor-based pagination
- **`create_foxnose_tool`** — search tool for LLM agents

## Installation

```bash
pip install langchain-foxnose
```

Requires Python 3.10+, `foxnose-sdk>=0.8.0`, and `langchain-core>=1.0`.

## Quick Start

```python
from foxnose_sdk.flux import FluxClient
from foxnose_sdk.auth import SimpleKeyAuth
from langchain_foxnose import FoxNoseRetriever

# Create a FoxNose Flux client
client = FluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)

# Create the retriever
retriever = FoxNoseRetriever(
    client=client,
    collection_path="knowledge-base",
    page_content_field="body",
    search_mode="hybrid",
    top_k=5,
)

# Use it
docs = retriever.invoke("How do I reset my password?")
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
```

> **Note (0.4.0):** The `folder_path` kwarg on `FoxNoseRetriever`, `FoxNoseLoader`,
> and `create_foxnose_tool` is deprecated in favor of `collection_path`. The
> legacy kwarg still works but emits a `DeprecationWarning`; it will be removed
> in 1.0. Requires `foxnose-sdk>=0.8.0`.

## Features

- **All search modes**: text, vector, hybrid, and vector-boosted search
- **Custom embeddings**: bring your own LangChain `Embeddings` model or pre-computed vectors
- **Bulk document loading**: cursor-based pagination with lazy loading for large collections
- **Document writing**: publish `Document` objects into a collection with external-id deduplication
- **Agent-ready search tool**: wrap any retriever as a tool for LLM agents
- **Flexible content mapping**: single field, multiple fields, or custom mapper function
- **Metadata control**: whitelist, blacklist, or include system metadata
- **Native async**: uses `AsyncFluxClient` for true async when available
- **Structured filtering**: pass FoxNose `where` filters for precise retrieval
- **Server-side truncation**: cap `text` field length with `truncate_text` instead of shipping whole documents
- **Full configuration**: search fields, thresholds, hybrid weights, sort, and more

## Search Modes

```python
# Pure vector (semantic) search
retriever = FoxNoseRetriever(
    client=client,
    collection_path="articles",
    page_content_field="body",
    search_mode="vector",
)

# Hybrid search (text + vector)
retriever = FoxNoseRetriever(
    client=client,
    collection_path="articles",
    page_content_field="body",
    search_mode="hybrid",
    hybrid_config={"vector_weight": 0.6, "text_weight": 0.4},
)

# Text search with vector boost
retriever = FoxNoseRetriever(
    client=client,
    collection_path="articles",
    page_content_field="body",
    search_mode="vector_boosted",
    vector_boost_config={"boost_factor": 1.3},
)
```

## Custom Embeddings

Use your own embedding model for vector search:

```python
from langchain_openai import OpenAIEmbeddings

retriever = FoxNoseRetriever(
    client=client,
    collection_path="articles",
    page_content_field="body",
    search_mode="vector",
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    vector_field="embedding",
)
```

Or pass a pre-computed vector directly:

```python
retriever = FoxNoseRetriever(
    client=client,
    collection_path="articles",
    page_content_field="body",
    search_mode="vector",
    query_vector=[0.1, 0.2, ...],
    vector_field="embedding",
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
            ]
        }
    },
)
```

## Document Loader

`FoxNoseLoader` iterates over all resources in a collection using cursor-based pagination. Use it to bulk-load documents for indexing, batch processing, or seeding a local vector store.

```python
from langchain_foxnose import FoxNoseLoader

loader = FoxNoseLoader(
    client=client,
    collection_path="knowledge-base",
    page_content_field="body",
    batch_size=50,
)

# Load all documents at once
docs = loader.load()

# Or iterate lazily for large collections
for doc in loader.lazy_load():
    print(doc.metadata.get("key"), doc.page_content[:100])
```

## Document Writer

`FoxNoseWriter` publishes `Document` objects into a collection. Requires a Flux
key with write access.

```python
from langchain_core.documents import Document
from langchain_foxnose import FoxNoseBatchWriteError, FoxNoseWriter

writer = FoxNoseWriter(
    client=client,
    collection_path="knowledge-base",
    page_content_field="body",
    external_id_key="source_id",   # metadata key used for deduplication
)

try:
    keys = writer.add_documents([
        Document(
            page_content="FoxNose is the knowledge layer for RAG.",
            metadata={"title": "What is FoxNose?", "source_id": "docs/intro"},
        ),
    ])
except FoxNoseBatchWriteError as exc:
    # exc.written_keys     -> written, NOT rolled back (Flux has no delete)
    # exc.failed_index     -> outcome UNKNOWN, re-read before retrying
    # exc.pending_indexes  -> guaranteed not attempted
    # exc.cause            -> the underlying typed SDK error; branch on this
    raise

# A full-document replace, not a merge:
writer.update_document(keys[0], Document(page_content="Updated.", metadata={...}))
```

Batches are written sequentially and stop at the first failure — there is no
concurrency option, because overlapping non-idempotent writes that cannot be
deleted make it impossible to report what was attempted. See the
[writer guide](https://langchain-foxnose.readthedocs.io/en/latest/writer/).

## Agent Tool

`create_foxnose_tool` wraps a retriever as a LangChain tool that LLM agents can call.

```python
from langchain_foxnose import create_foxnose_tool

tool = create_foxnose_tool(
    client=client,
    collection_path="knowledge-base",
    page_content_field="body",
    name="kb_search",
    description="Search the knowledge base for relevant information.",
    search_mode="hybrid",
    top_k=5,
)

# Use directly
result = tool.invoke("How do I reset my password?")

# Or plug into a LangChain agent
# from langchain.agents import create_agent
# agent = create_agent(model="openai:gpt-4o", tools=[tool])
```

## Async Usage

```python
from foxnose_sdk.flux import AsyncFluxClient

async_client = AsyncFluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)

retriever = FoxNoseRetriever(
    async_client=async_client,
    collection_path="knowledge-base",
    page_content_field="body",
)

docs = await retriever.ainvoke("search query")
```

## Documentation

- [Getting Started](https://langchain-foxnose.readthedocs.io/en/latest/getting-started/)
- [Retriever](https://langchain-foxnose.readthedocs.io/en/latest/retriever/)
- [Document Loader](https://langchain-foxnose.readthedocs.io/en/latest/loader/)
- [Document Writer](https://langchain-foxnose.readthedocs.io/en/latest/writer/)
- [Search Tool](https://langchain-foxnose.readthedocs.io/en/latest/tool/)
- [Configuration](https://langchain-foxnose.readthedocs.io/en/latest/configuration/)
- [Examples](https://langchain-foxnose.readthedocs.io/en/latest/examples/)
- [API Reference](https://langchain-foxnose.readthedocs.io/en/latest/api-reference/)
- [FoxNose Documentation](https://foxnose.net/docs?utm_source=github&utm_medium=repository&utm_campaign=langchain-foxnose)

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.

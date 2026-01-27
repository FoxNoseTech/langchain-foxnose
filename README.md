# langchain-foxnose

[![PyPI version](https://img.shields.io/pypi/v/langchain-foxnose.svg)](https://pypi.org/project/langchain-foxnose/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-foxnose.svg)](https://pypi.org/project/langchain-foxnose/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/FoxNoseTech/langchain-foxnose/actions/workflows/ci.yml/badge.svg)](https://github.com/FoxNoseTech/langchain-foxnose/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/FoxNoseTech/langchain-foxnose/branch/main/graph/badge.svg)](https://codecov.io/gh/FoxNoseTech/langchain-foxnose)

LangChain integration for [FoxNose](https://foxnose.net) — the serverless knowledge platform purpose-built as the knowledge layer for RAG and AI agents.

## Installation

```bash
pip install langchain-foxnose
```

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
    folder_path="knowledge-base",
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

## Features

- **All search modes**: text, vector, hybrid, and vector-boosted search
- **Flexible content mapping**: single field, multiple fields, or custom mapper function
- **Metadata control**: whitelist, blacklist, or include system metadata
- **Native async**: uses `AsyncFluxClient` for true async when available
- **Structured filtering**: pass FoxNose `where` filters for precise retrieval
- **Full configuration**: search fields, thresholds, hybrid weights, sort, and more

## Search Modes

```python
# Pure vector (semantic) search
retriever = FoxNoseRetriever(
    client=client,
    folder_path="articles",
    page_content_field="body",
    search_mode="vector",
)

# Hybrid search (text + vector)
retriever = FoxNoseRetriever(
    client=client,
    folder_path="articles",
    page_content_field="body",
    search_mode="hybrid",
    hybrid_config={"vector_weight": 0.6, "text_weight": 0.4},
)

# Text search with vector boost
retriever = FoxNoseRetriever(
    client=client,
    folder_path="articles",
    page_content_field="body",
    search_mode="vector_boosted",
    vector_boost_config={"boost_factor": 1.3},
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
            ]
        }
    },
)
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
    folder_path="knowledge-base",
    page_content_field="body",
)

docs = await retriever.ainvoke("search query")
```

## Documentation

- [Getting Started](https://langchain-foxnose.readthedocs.io/getting-started/)
- [Configuration](https://langchain-foxnose.readthedocs.io/configuration/)
- [Examples](https://langchain-foxnose.readthedocs.io/examples/)
- [API Reference](https://langchain-foxnose.readthedocs.io/api-reference/)

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.

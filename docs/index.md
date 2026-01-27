# langchain-foxnose

LangChain integration for [FoxNose](https://foxnose.net?utm_source=readthedocs&utm_medium=documentation&utm_campaign=langchain-foxnose) — the serverless knowledge platform purpose-built as the knowledge layer for RAG and AI agents.

## Why langchain-foxnose?

FoxNose eliminates the "DIY RAG problem" where developers need to stitch together a primary database + full-text search engine + vector database + ETL scripts. With FoxNose, you get:

- **No separate vector DB** — embeddings are automatic when fields are marked `vectorizable`
- **Single `_search` endpoint** — handles text + vector + hybrid + filters in one request
- **No ETL/sync** — content changes auto-update embeddings
- **Enterprise features** — environments, localization, versioning, RBAC out of the box

This package provides a LangChain `BaseRetriever` implementation that wraps the FoxNose Flux API, letting you use FoxNose as the knowledge retrieval layer in any LangChain pipeline.

## Quick Start

```bash
pip install langchain-foxnose
```

```python
from foxnose_sdk.flux import FluxClient
from foxnose_sdk.auth import SimpleKeyAuth
from langchain_foxnose import FoxNoseRetriever

client = FluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)

retriever = FoxNoseRetriever(
    client=client,
    folder_path="knowledge-base",
    page_content_field="body",
    search_mode="hybrid",
    top_k=5,
)

docs = retriever.invoke("How do I reset my password?")
```

## Next Steps

- [Getting Started](getting-started.md) — installation, prerequisites, first retriever
- [Retriever](retriever.md) — deep dive into `FoxNoseRetriever`
- [Configuration](configuration.md) — all parameters explained
- [Examples](examples.md) — common patterns and use cases
- [API Reference](api-reference.md) — auto-generated from source
- [FoxNose Documentation](https://foxnose.net/docs?utm_source=readthedocs&utm_medium=documentation&utm_campaign=langchain-foxnose) — main FoxNose platform docs

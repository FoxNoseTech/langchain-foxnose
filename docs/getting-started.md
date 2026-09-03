# Getting Started

## Prerequisites

- Python >= 3.9
- A [FoxNose](https://foxnose.net?utm_source=readthedocs&utm_medium=documentation&utm_campaign=langchain-foxnose) workspace with at least one folder containing vectorized content
- A Flux API key (public + secret)

See the [FoxNose documentation](https://foxnose.net/docs?utm_source=readthedocs&utm_medium=documentation&utm_campaign=langchain-foxnose) for workspace setup and API key creation.

## Installation

```bash
pip install langchain-foxnose
```

This installs `langchain-foxnose` along with its dependencies: `foxnose-sdk` and `langchain-core`.

## Your First Retriever

### 1. Create a Flux client

```python
from foxnose_sdk.flux import FluxClient
from foxnose_sdk.auth import SimpleKeyAuth

client = FluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)
```

Replace `<env_key>` with your environment key, and provide your API credentials.

### 2. Create a retriever

```python
from langchain_foxnose import FoxNoseRetriever

retriever = FoxNoseRetriever(
    client=client,
    collection_path="knowledge-base",
    page_content_field="body",
    search_mode="hybrid",
    top_k=5,
)
```

- `collection_path` — the FoxNose collection connected to your Flux API
- `page_content_field` — which `data` field becomes the document's `page_content`
- `search_mode` — `"text"`, `"vector"`, `"hybrid"`, or `"vector_boosted"`
- `top_k` — how many results to return

### 3. Retrieve documents

```python
docs = retriever.invoke("How do I reset my password?")
for doc in docs:
    print(doc.page_content[:100])
    print(doc.metadata)
```

### 4. Use with a LangChain agent

Wrap the retriever as a tool and hand it to an agent:

```python
from langchain.agents import create_agent

from langchain_foxnose import create_foxnose_tool

tool = create_foxnose_tool(
    client=client,
    collection_path="knowledge-base",
    page_content_field="body",
)
agent = create_agent(model="openai:gpt-4o", tools=[tool])

result = agent.invoke(
    {"messages": [{"role": "user", "content": "How do I reset my password?"}]}
)
print(result["messages"][-1].content)
```

This needs the `langchain` package, which `langchain-foxnose` does not depend
on — it only requires `langchain-core`, so the agent framework stays your
choice. Install it with `pip install langchain langchain-openai`.

!!! note "Migrating from LangChain 0.3"

    `RetrievalQA` and the rest of `langchain.chains` were removed in LangChain
    1.0 — the module no longer exists. `langchain.agents.create_agent` is the
    replacement, and it takes `{"messages": [...]}` rather than
    `{"query": ...}`. The legacy chains live on in the separate
    `langchain-classic` package, but they are deprecated there with a declared
    removal in 2.0, so new code should use an agent.

## Convenience Constructor

If you prefer not to construct the client yourself:

```python
from foxnose_sdk.auth import SimpleKeyAuth
from langchain_foxnose import FoxNoseRetriever

retriever = FoxNoseRetriever.from_client_params(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
    collection_path="knowledge-base",
    page_content_field="body",
)
```

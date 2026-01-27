# Getting Started

## Prerequisites

- Python >= 3.9
- A FoxNose workspace with at least one folder containing vectorized content
- A Flux API key (public + secret)

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
    folder_path="knowledge-base",
    page_content_field="body",
    search_mode="hybrid",
    top_k=5,
)
```

- `folder_path` — the FoxNose folder connected to your Flux API
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

### 4. Use in a LangChain chain

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=retriever,
    return_source_documents=True,
)
result = qa.invoke({"query": "How do I reset my password?"})
print(result["result"])
```

## Convenience Constructor

If you prefer not to construct the client yourself:

```python
from foxnose_sdk.auth import SimpleKeyAuth
from langchain_foxnose import FoxNoseRetriever

retriever = FoxNoseRetriever.from_client_params(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
    folder_path="knowledge-base",
    page_content_field="body",
)
```

# Retriever

`FoxNoseRetriever` is a LangChain `BaseRetriever` implementation that queries the FoxNose Flux `_search` endpoint.

## Architecture

```
FoxNoseRetriever
  ├── _execute_search()           → dispatches to the appropriate SDK method
  │   ├── client.vector_search()       (vector mode, auto embeddings)
  │   ├── client.vector_field_search() (vector mode, custom embeddings)
  │   ├── client.hybrid_search()       (hybrid mode)
  │   ├── client.boosted_search()      (vector_boosted mode)
  │   └── client.search()             (text mode)
  └── map_results_to_documents()  → converts results to LangChain Documents
```

The retriever uses SDK v0.5.0 convenience methods for validated, type-safe search requests.

## Content Mapping

FoxNose returns structured results with `_sys` (system metadata) and `data` (your fields). You must tell the retriever which field(s) become `page_content`.

### Single field

```python
retriever = FoxNoseRetriever(
    client=client,
    folder_path="articles",
    page_content_field="body",
)
```

### Multiple fields (concatenated)

```python
retriever = FoxNoseRetriever(
    client=client,
    folder_path="articles",
    page_content_fields=["title", "body"],
    page_content_separator="\n\n",  # default
)
```

### Custom mapper

For full control, pass a callable that receives the raw result dict:

```python
retriever = FoxNoseRetriever(
    client=client,
    folder_path="articles",
    page_content_mapper=lambda result: (
        f"# {result['data']['title']}\n\n{result['data']['body']}"
    ),
)
```

## Metadata

By default, metadata includes:

- `_sys` fields: `key`, `folder`, `created_at`, `updated_at`
- All `data` fields except those used for `page_content`

### Whitelist

```python
retriever = FoxNoseRetriever(
    ...,
    metadata_fields=["title", "category"],  # only these data fields
)
```

### Blacklist

```python
retriever = FoxNoseRetriever(
    ...,
    exclude_metadata_fields=["internal_notes"],  # exclude these
)
```

### Disable system metadata

```python
retriever = FoxNoseRetriever(
    ...,
    include_sys_metadata=False,
)
```

## Custom Embeddings (Vector Field Search)

When you have your own embedding model or pre-computed vectors, use `vector_field` together with `embeddings` or `query_vector` to search via the SDK's `vector_field_search()` method.

### With a LangChain Embeddings model

The retriever converts the query text into a vector at query time:

```python
from langchain_openai import OpenAIEmbeddings

retriever = FoxNoseRetriever(
    client=client,
    folder_path="articles",
    page_content_field="body",
    search_mode="vector",
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    vector_field="embedding",       # field name in FoxNose
    similarity_threshold=0.75,
)

docs = retriever.invoke("How do I reset my password?")
```

!!! warning
    The query text is sent to the embedding provider (e.g. OpenAI) on every invocation.

### With a static query vector

If you already have a vector, pass it directly:

```python
retriever = FoxNoseRetriever(
    client=client,
    folder_path="articles",
    page_content_field="body",
    search_mode="vector",
    query_vector=[0.1, 0.2, ...],   # your pre-computed vector
    vector_field="embedding",
)
```

### With vector-boosted mode

Custom embeddings also work in `vector_boosted` mode. The retriever sends both text and vector, using the vector for similarity boosting:

```python
retriever = FoxNoseRetriever(
    client=client,
    folder_path="articles",
    page_content_field="body",
    search_mode="vector_boosted",
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    vector_field="embedding",
    vector_boost_config={"boost_factor": 1.5},
)
```

### Validation rules

- `embeddings` and `query_vector` are mutually exclusive
- `vector_field` is required when either is set
- `vector_field` and `vector_fields` are mutually exclusive (`vector_field` for custom embeddings, `vector_fields` for auto-generated)
- Custom embeddings are only supported in `vector` and `vector_boosted` modes
- `query_vector` must be non-empty with finite values (no NaN/Inf)

## Sync vs Async

### Sync (default)

```python
from foxnose_sdk.flux import FluxClient

client = FluxClient(...)
retriever = FoxNoseRetriever(client=client, ...)
docs = retriever.invoke("query")
```

### Native async

```python
from foxnose_sdk.flux import AsyncFluxClient

async_client = AsyncFluxClient(...)
retriever = FoxNoseRetriever(async_client=async_client, ...)
docs = await retriever.ainvoke("query")
```

### Fallback

When only a sync `client` is provided, `ainvoke()` falls back to running the sync search in an executor (the default LangChain behaviour).

### Both clients

You can provide both. Sync calls use `client`, async calls use `async_client`:

```python
retriever = FoxNoseRetriever(
    client=sync_client,
    async_client=async_client,
    ...,
)
```

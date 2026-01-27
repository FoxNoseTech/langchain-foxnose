# Retriever

`FoxNoseRetriever` is a LangChain `BaseRetriever` implementation that queries the FoxNose Flux `_search` endpoint.

## Architecture

```
FoxNoseRetriever
  ├── build_search_body()   → constructs the _search request body
  ├── FluxClient.search()   → sends the request to FoxNose
  └── map_results_to_documents() → converts results to LangChain Documents
```

Each component is a pure function (except the client call), making the retriever easy to test and extend.

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

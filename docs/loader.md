# Document Loader

`FoxNoseLoader` is a LangChain `BaseLoader` that iterates over all resources in a FoxNose folder using cursor-based pagination.

Use it when you need to **bulk-load** every document in a folder — for example to seed a local vector store, build an index, or run batch processing.

## Quick Start

```python
from foxnose_sdk.flux import FluxClient
from foxnose_sdk.auth import SimpleKeyAuth
from langchain_foxnose import FoxNoseLoader

client = FluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)

loader = FoxNoseLoader(
    client=client,
    folder_path="knowledge-base",
    page_content_field="body",
)

docs = loader.load()  # loads all documents
```

## Lazy Loading

For large folders, use `lazy_load()` to stream documents without holding everything in memory:

```python
for doc in loader.lazy_load():
    print(doc.metadata["key"], doc.page_content[:100])
```

## Async Loading

```python
from foxnose_sdk.flux import AsyncFluxClient

async_client = AsyncFluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)

loader = FoxNoseLoader(
    async_client=async_client,
    folder_path="knowledge-base",
    page_content_field="body",
)

async for doc in loader.alazy_load():
    print(doc.metadata["key"])
```

## Filtering & Sorting

Pass query parameters via `params` to filter or sort results:

```python
loader = FoxNoseLoader(
    client=client,
    folder_path="articles",
    page_content_field="body",
    params={
        "where": {"status__eq": "published"},
        "sort": "-created_at",
    },
)
```

## Pagination

The loader automatically handles cursor-based pagination. Control the page size with `batch_size`:

```python
loader = FoxNoseLoader(
    client=client,
    folder_path="articles",
    page_content_field="body",
    batch_size=50,  # fetch 50 resources per page (default: 100)
)
```

## Content Mapping

Content mapping works the same way as in `FoxNoseRetriever`. See [Configuration](configuration.md) for details.

| Strategy | Parameter | Description |
|----------|-----------|-------------|
| Single field | `page_content_field` | One `data` field becomes `page_content` |
| Multi field | `page_content_fields` | Multiple fields concatenated with `page_content_separator` |
| Custom mapper | `page_content_mapper` | Callable `(result_dict) -> str` for full control |

## Metadata

| Parameter | Description |
|-----------|-------------|
| `metadata_fields` | Whitelist of `data` fields to include |
| `exclude_metadata_fields` | Blacklist of `data` fields to exclude |
| `include_sys_metadata` | Include `_sys` fields (default: `True`) |

## API Reference

::: langchain_foxnose.loaders.FoxNoseLoader

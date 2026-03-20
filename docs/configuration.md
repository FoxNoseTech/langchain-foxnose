# Configuration

All parameters for `FoxNoseRetriever`.

## Client

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `client` | `FluxClient` | One of `client`/`async_client` | Synchronous Flux client |
| `async_client` | `AsyncFluxClient` | One of `client`/`async_client` | Async Flux client |

## Folder

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `folder_path` | `str` | Yes | — | FoxNose folder path (e.g. `"knowledge-base"`) |

## Content Mapping

Exactly one of these must be set:

| Parameter | Type | Description |
|-----------|------|-------------|
| `page_content_field` | `str` | Single `data` field for `page_content` |
| `page_content_fields` | `list[str]` | Multiple fields concatenated (must be non-empty) |
| `page_content_mapper` | `Callable` | Custom `(result) -> str` function |

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page_content_separator` | `str` | `"\n\n"` | Separator for `page_content_fields` |

## Metadata

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metadata_fields` | `list[str]` | `None` | Whitelist of `data` fields for metadata |
| `exclude_metadata_fields` | `list[str]` | `None` | Blacklist of `data` fields |
| `include_sys_metadata` | `bool` | `True` | Include `_sys` fields (key, folder, created_at, updated_at) |

!!! warning
    `metadata_fields` and `exclude_metadata_fields` are mutually exclusive. Setting both raises a validation error.

## Search

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_mode` | `str` | `"hybrid"` | `"text"`, `"vector"`, `"hybrid"`, or `"vector_boosted"` |
| `top_k` | `int` | `5` | Maximum results to return (must be >= 1) |
| `search_fields` | `list[str]` | `None` | Fields for text search (`find_text.fields`) |
| `text_threshold` | `float` | `None` | Typo tolerance (`find_text.threshold`, 0-1) |
| `vector_fields` | `list[str]` | `None` | Fields for vector search (`vector_search.fields`) |
| `similarity_threshold` | `float` | `None` | Minimum cosine similarity (0-1) |
| `where` | `dict` | `None` | Persistent structured filter |
| `hybrid_config` | `dict` | `None` | Hybrid mode config (`vector_weight`, `text_weight`, `rerank_results`) |
| `vector_boost_config` | `dict` | `None` | Boost config (`boost_factor`, `similarity_threshold`, `max_boost_results`) |
| `sort` | `list[str]` | `None` | Sort fields (prefix `-` for descending) |
| `search_kwargs` | `dict` | `{}` | Extra params passed to SDK methods (see below) |

!!! note
    Known keys in `search_kwargs` like `limit` and `offset` are extracted as named
    SDK method parameters. The rest are passed through `**extra_body`.

!!! warning
    `search_kwargs` must not contain keys that conflict with `SearchRequest` fields:
    `search_mode`, `vector_search`, `vector_field_search`, `hybrid_config`,
    `vector_boost_config`, `find_text`, `find_phrase`.

!!! note
    `text_threshold` and `similarity_threshold` are validated to be in the range 0-1.

## Custom Embeddings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embeddings` | `Embeddings` | `None` | LangChain Embeddings model for converting query text to vectors |
| `query_vector` | `list[float]` | `None` | Pre-computed static query vector |
| `vector_field` | `str` | `None` | Field name for `vector_field_search` (required when `embeddings` or `query_vector` is set) |

!!! warning
    `embeddings` and `query_vector` are mutually exclusive.
    `vector_field` and `vector_fields` are mutually exclusive.
    Custom embeddings are only supported in `vector` and `vector_boosted` modes.

!!! warning
    When using `embeddings`, the query text may be sent to a third-party service (e.g. OpenAI)
    depending on the Embeddings implementation.

## Search Modes

| Mode | Text Search | Vector Search | Use Case |
|------|-------------|---------------|----------|
| `text` | Required | Not allowed | Keyword/phrase search |
| `vector` | Not allowed | Required | Pure semantic search |
| `hybrid` | Required | Required | Blended text + vector with weights |
| `vector_boosted` | Required | Required | Text results boosted by vector similarity |

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
| `search_kwargs` | `dict` | `{}` | Extra params merged into body (overrides) |

!!! note
    Overriding `"limit"` via `search_kwargs` does **not** update `vector_search.top_k`.
    Only the outer `limit` in the request body changes.

!!! warning
    `search_kwargs` must not contain `"search_mode"`. Set `search_mode` directly instead.
    Overriding it via kwargs would create an inconsistent request body.

!!! note
    `text_threshold` and `similarity_threshold` are validated to be in the range 0-1.

## Search Modes

| Mode | Text Search | Vector Search | Use Case |
|------|-------------|---------------|----------|
| `text` | Required | Not allowed | Keyword/phrase search |
| `vector` | Not allowed | Required | Pure semantic search |
| `hybrid` | Required | Required | Blended text + vector with weights |
| `vector_boosted` | Required | Required | Text results boosted by vector similarity |

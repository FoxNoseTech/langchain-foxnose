# Changelog

## 0.3.0 (2026-03-20)

- **Vector field search with custom embeddings** — new `embeddings`, `query_vector`, and `vector_field` parameters on `FoxNoseRetriever` allow using custom pre-computed embedding vectors or LangChain `Embeddings` models for semantic search via the SDK's `vector_field_search()` method
- **SDK convenience methods** — the retriever now uses SDK v0.5.0 convenience methods (`vector_search()`, `hybrid_search()`, `boosted_search()`, `vector_field_search()`) instead of manually building request bodies
- **Strict config validation** — `hybrid_config` and `vector_boost_config` dicts are now validated against strict models that reject unknown keys (typos are caught at init time)
- **search_kwargs improvements** — known keys like `limit` and `offset` are extracted as named SDK method parameters; conflicting keys (e.g. `search_mode`, `vector_search`) are rejected at validation time
- **Deprecation** — `build_search_body()` is deprecated and will be removed in v0.4.0
- Require `foxnose-sdk>=0.5.0`

## 0.2.1 (2026-03-18)

- Align package metadata versioning so `langchain_foxnose.__version__` matches the published package version
- Require `foxnose-sdk>=0.4.2` for the secure-auth signing fix

## 0.1.0 (2026-01-27)

Initial release.

- `FoxNoseRetriever` — LangChain `BaseRetriever` backed by FoxNose Flux search
- Support for all search modes: text, vector, hybrid, and vector-boosted
- Flexible content mapping: single field, multiple fields, or custom mapper
- Metadata control: whitelist, blacklist, system metadata toggle
- Native async support via `AsyncFluxClient`
- Structured filtering via `where` parameter
- Convenience `from_client_params()` constructor

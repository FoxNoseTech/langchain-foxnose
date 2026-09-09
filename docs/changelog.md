# Changelog

## 0.4.0 (2026-09-09)

### Changed

- **Renamed `folder_path` → `collection_path`** on `FoxNoseRetriever`,
  `FoxNoseLoader`, and `create_foxnose_tool`, following the SDK's
  Folder → Collection rename. The legacy `folder_path` kwarg is still accepted
  with a one-shot `DeprecationWarning` and will be removed in 1.0; passing both
  raises `ValueError`.
- **BREAKING: requires Python 3.10+.** `langchain-core` 1.x declares
  `Requires-Python >=3.10`, so Python 3.9 is no longer supported. Python 3.9
  users stay on 0.3.1.
- **BREAKING: requires `langchain-core>=1.0,<2.0`.** Support for the 0.3.x line
  is dropped. The integration's public behaviour is unchanged — the
  `langchain-core` surface it uses (`BaseRetriever`, `BaseLoader`,
  `create_retriever_tool`) is identical across 0.3.x and 1.x.
- Require `foxnose-sdk>=0.8.1`, for the `data_validation_error` mapping a Flux
  write depends on: before it, a schema violation arrived as a plain
  `FoxnoseAPIError` and no `except ContentValidationFailed` branch could fire.

### Added

- **`FoxNoseWriter`** — write LangChain `Document` objects into a FoxNose
  collection via Flux `create_resource()` / `update_resource()`, with
  `add_documents()`, `update_document()`, and async variants. Supports
  `external_id_key` deduplication, a custom `document_mapper`, and metadata
  whitelisting/blacklisting. Batches are written sequentially and stop at the
  first failure; there is deliberately no concurrency option, because
  overlapping non-idempotent writes cannot be reported on honestly.
- **`FoxNoseBatchWriteError`** — raised when a document batch fails part-way
  through, carrying `written_keys` (written, **not rolled back** — Flux has no
  delete endpoint), `failed_index` (outcome **unknown**: a 422/409 wrote
  nothing, but a 502 or timeout may have written it), `pending_indexes`
  (guaranteed not attempted), `total`, and `cause` (the underlying typed SDK
  error — branch on that, not on a second `except`). A local mapping error is
  raised before any request, so it writes nothing at all.
- **`truncate_text` and `query_params`** on `FoxNoseRetriever`, and
  `truncate_text` on `FoxNoseLoader` — forward query-string parameters to the
  FoxNose API to cap the length of `text`-typed fields server-side.

### Removed

- `build_search_body()` (module `langchain_foxnose._search`), deprecated since
  0.3.0.

### Fixed

- **`FoxNoseLoader` could paginate forever.** FoxNose returns the `next` field
  as a full URL, not as the opaque token the `next` query parameter accepts.
  Feeding the URL back meant the backend could not parse it, silently answered
  with page one again and returned the same `next` — `load()` re-fetched the
  first page indefinitely (17k requests in 30 seconds against a live backend).
  Cursors are now reduced to their token, and each is followed at most once, so
  a backend that cycles (`A -> B -> A`) terminates instead of spinning.
- **`top_k` did not limit results in `hybrid` and `vector_boosted` modes.** It
  was forwarded only as the vector-side candidate count, leaving the page size
  at the backend default, so a retriever built with `top_k=3` could return
  every matching document. It now caps the results in every mode, as
  documented; an explicit `search_kwargs={"limit": ...}` still takes
  precedence.
- **`py.typed` is now shipped inside the package.** The marker sat at the
  repository root, where PEP 561 does not look for it, so it never reached the
  wheel: every downstream project got `missing py.typed marker` from mypy and
  fell back to `Any` for the whole package, despite the source being fully
  annotated.
- `search_kwargs={"truncate_text": ...}` used to reach the request body, where
  `foxnose-sdk` 0.8.0 rejects it. Query-string keys are now rejected at
  validation time with a message pointing at the new parameters.
- **Documentation and examples migrated to the LangChain 1.x agent pattern.**
  `docs/getting-started.md` and `docs/examples.md` imported `RetrievalQA` from
  `langchain.chains`, a module that **no longer exists** in LangChain 1.x
  (`ModuleNotFoundError`) — those examples were unrunnable. They now use
  `langchain.agents.create_agent`, and source documents come from the tool's
  `content_and_artifact` response rather than the removed
  `result["source_documents"]`. `docs/tool.md`, `examples/agent_tool.py` and
  the README moved off the deprecated `langgraph.prebuilt.create_react_agent`.
- Documentation and examples now use `collection_path` rather than the
  deprecated `folder_path`, so following the docs no longer triggers a
  `DeprecationWarning`.
- `mypy`'s `python_version` pin was dropped: pinning it to the 3.10 floor made
  mypy parse third-party stubs under 3.10 rules, and numpy >= 2.3 (pulled in
  transitively by `langchain-tests`) ships PEP 695 stubs that are 3.12-only
  syntax — so `mypy src/` failed in any environment with both extras
  installed, which is what `make install` creates. The floor is enforced by
  ruff's `target-version` and by the CI matrix actually running on 3.10.

## 0.3.1 (2026-03-23)

### Fixed

- `limit` now defaults to `top_k` in the vector search modes.
- Added a runtime `k` alias for `top_k` on `invoke()` / `ainvoke()`.
- Added a CI guard that fails fast when integration-test secrets are missing.

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

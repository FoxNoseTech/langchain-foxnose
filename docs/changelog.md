# Changelog

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

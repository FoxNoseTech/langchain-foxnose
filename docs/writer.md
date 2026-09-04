# Document Writer

`FoxNoseWriter` writes LangChain `Document` objects into a FoxNose collection
through the Flux write API. Each document becomes one FoxNose resource,
published immediately.

It is the write counterpart to [`FoxNoseRetriever`](retriever.md) and
[`FoxNoseLoader`](loader.md). Requires `foxnose-sdk>=0.8.0`.

## Quick Start

```python
from foxnose_sdk.flux import FluxClient
from foxnose_sdk.auth import SimpleKeyAuth
from langchain_core.documents import Document
from langchain_foxnose import FoxNoseWriter

client = FluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)

writer = FoxNoseWriter(
    client=client,
    collection_path="knowledge-base",
    page_content_field="body",
)

keys = writer.add_documents([
    Document(page_content="FoxNose is the knowledge layer for RAG.",
             metadata={"title": "What is FoxNose?"}),
])
print(keys)  # ['<resource_key>']
```

`page_content` goes into `page_content_field`; everything in `metadata`
becomes a sibling `data` field.

## Convenience Constructor

```python
writer = FoxNoseWriter.from_client_params(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
    collection_path="knowledge-base",
    page_content_field="body",
)
```

## Updating

`update_document` takes the **internal** resource key — the value
`add_documents` returned, or `doc.metadata["key"]` on a document you read back.

```python
revision = writer.update_document(keys[0], Document(
    page_content="FoxNose is the knowledge layer for RAG. Updated.",
    metadata={"title": "What is FoxNose?"},
))
```

!!! warning "Update is a full replace, not a merge"

    Fields absent from the mapped `data` are **removed** from the stored
    resource. If you want to change one field, read the document first, edit
    it, and write the whole thing back — see [Round-tripping](#round-tripping).

## Async

```python
import asyncio
from foxnose_sdk.flux import AsyncFluxClient

async def main():
    async_client = AsyncFluxClient(
        base_url="https://<env_key>.fxns.io",
        api_prefix="my_api",
        auth=SimpleKeyAuth("pk", "sk"),
    )
    writer = FoxNoseWriter(
        async_client=async_client,
        collection_path="knowledge-base",
        page_content_field="body",
    )
    keys = await writer.aadd_documents([Document(page_content="Hello")])
    await writer.aupdate_document(keys[0], Document(page_content="Hello again"))
    await async_client.aclose()

asyncio.run(main())
```

!!! note "There is no concurrency option, on purpose"

    Both `add_documents` and `aadd_documents` write **sequentially** and stop
    at the first failure. Flux writes are non-idempotent, the SDK never retries
    them, and Flux has no delete endpoint — so overlapping them would make it
    impossible to report which documents were attempted. If you want
    concurrency, chunk the input and drive `aadd_documents` yourself, accepting
    that ambiguity.

## Deduplication with `external_id_key`

Point `external_id_key` at a metadata key holding your own stable identifier.
Its value is sent as the resource `key` and is **not** written into `data`:

```python
writer = FoxNoseWriter(
    client=client,
    collection_path="knowledge-base",
    page_content_field="body",
    external_id_key="source_id",
)

writer.add_documents([
    Document(page_content="…", metadata={"title": "Hybrid search",
                                         "source_id": "docs/hybrid-search"}),
])
```

Reusing a value raises `ExternalIdConflict`. A document without that metadata
key is written without an external id — a collection may legitimately hold a
mix. The value must be a `str` or `int`; anything else raises `TypeError`
naming the document index, rather than silently turning an object into a key
you could never reproduce.

`update_document` ignores `external_id_key`: `update_resource` addresses a
resource by its internal key and takes no external one.

## Controlling what gets written

| Parameter | Default | Description |
|-----------|---------|-------------|
| `page_content_field` | — | `data` field that receives `page_content` |
| `document_mapper` | `None` | Custom `(document) -> dict` for full control; mutually exclusive with `page_content_field` |
| `metadata_fields` | `None` | Whitelist of metadata keys to write |
| `exclude_metadata_fields` | `None` | Blacklist of metadata keys to skip |
| `include_sys_metadata` | `False` | Write the read path's `_sys`-derived keys (`key`, `folder`, `created_at`, `updated_at`) |
| `external_id_key` | `None` | Metadata key holding an external deduplication id |

Exactly one of `page_content_field` / `document_mapper` must be set, and
`metadata_fields` / `exclude_metadata_fields` are mutually exclusive.

A custom mapper takes over completely — every other mapping option is ignored:

```python
writer = FoxNoseWriter(
    client=client,
    collection_path="articles",
    document_mapper=lambda d: {
        "body": d.page_content,
        "title": d.metadata.get("title", "Untitled"),
        "word_count": len(d.page_content.split()),
    },
)
```

## Round-tripping

`FoxNoseLoader` and `FoxNoseRetriever` include `_sys` metadata by default, so a
loaded document carries `key`, `folder`, `created_at` and `updated_at`. None of
those are schema fields. The writer **strips them automatically**, so a
read-edit-write cycle just works:

```python
from langchain_foxnose import FoxNoseLoader, FoxNoseWriter

loader = FoxNoseLoader(
    client=client, collection_path="articles", page_content_field="body"
)
writer = FoxNoseWriter(
    client=client, collection_path="articles", page_content_field="body"
)

for doc in loader.load():
    doc.page_content = doc.page_content.strip()
    writer.update_document(doc.metadata["key"], doc)
```

Set `include_sys_metadata=True` on the writer only if your schema really does
have fields with those names.

## Errors

Every underlying error from `add_documents` / `aadd_documents` is wrapped in
`FoxNoseBatchWriteError`. **Branch on `exc.cause`** — a second `except` clause
after it is dead code:

```python
from foxnose_sdk.errors import ExternalIdConflict, FoxnoseAPIError
from langchain_foxnose import FoxNoseBatchWriteError

try:
    keys = writer.add_documents(documents)
except FoxNoseBatchWriteError as exc:
    print(f"Written, not rolled back: {exc.written_keys}")
    print(f"Unknown outcome:          document {exc.failed_index}")
    print(f"Not attempted:            {exc.pending_indexes}")

    cause = exc.cause
    if isinstance(cause, ExternalIdConflict):
        print("  that external id already exists")
    elif isinstance(cause, FoxnoseAPIError) and cause.status_code == 422:
        print(f"  schema rejected the document: {cause.detail}")
    raise
```

!!! warning "Match a schema violation on the status code, not on `ContentValidationFailed`"

    `foxnose-sdk` maps only `(422, "content_validation_failed")` onto the typed
    `ContentValidationFailed`, but a **Flux write** that violates the schema
    comes back as `data_validation_error`, which is not mapped — so it arrives
    as a plain `FoxnoseAPIError` and an `isinstance(cause,
    ContentValidationFailed)` branch never fires. Verified against a live
    backend. Check `cause.status_code == 422` instead, and read `cause.detail`;
    if you also want the structured list, guard it:

    ```python
    errors = getattr(cause, "errors", None)  # only on ContentValidationFailed
    ```

The three ranges mean exactly this:

| Attribute | Guarantee |
|-----------|-----------|
| `written_keys` | Documents `0 … failed_index-1`. Written and **not rolled back** — Flux has no delete endpoint, so they stay. |
| `failed_index` | The document whose write raised. Its outcome is **unknown**, not "failed": a schema violation (422) or `ExternalIdConflict` (409) wrote nothing, but an `UpstreamError` (502) or a transport timeout may have written it. |
| `pending_indexes` | Documents after the failure. Guaranteed **not attempted** — this is what the sequential, stop-at-first-failure design buys you. |

!!! danger "Never blindly retry a failed write"

    For an `UpstreamError` or a timeout the write may already have landed.
    Re-read the resource with a GET to establish the real state before
    retrying, exactly as the SDK advises.

A **local** mapping error — a non-scalar external id, a metadata key colliding
with `page_content_field`, a raising `document_mapper` — is raised as a plain
`TypeError` / `ValueError` **before any request is sent**, so nothing is
written. `FoxNoseBatchWriteError` is reserved for writes that were actually
attempted.

Typed SDK errors you can meet directly from `update_document` (which writes one
resource and therefore does not wrap):

| Exception | HTTP | Meaning |
|-----------|------|---------|
| `CollectionNotWritable` | 403 | The collection's connection does not accept writes, or the key lacks write access |
| `ExternalIdConflict` | 409 | The supplied external id already identifies a resource |
| `ContentValidationFailed` | 422 | `data` failed the collection schema; see `.errors` / `.errors_truncated`. **A Flux write reports this as a plain `FoxnoseAPIError` with `error_code="data_validation_error"`** — see the warning above. |
| `UpstreamError` | 502 | The write could not be confirmed — outcome unknown |

## Requirements

Writes need a Flux key with write access, and the collection must be exposed on
the API with `create` / `update` in its allowed methods. An anonymous caller
gets 401; a key without the grant gets a generic 403.

!!! warning "Flux has no delete endpoint"

    Documents written by mistake cannot be removed through this integration.
    Use the Management API or the FoxNose dashboard.

## API Reference

::: langchain_foxnose.writers.FoxNoseWriter

::: langchain_foxnose.writers.FoxNoseBatchWriteError

"""Writer example — write LangChain Documents into a FoxNose collection.

Needs a Flux key with write access, and the collection must be exposed on the
API with `create` and `update` in its allowed methods.
"""

from foxnose_sdk.auth import SimpleKeyAuth
from foxnose_sdk.errors import ContentValidationFailed, ExternalIdConflict
from foxnose_sdk.flux import FluxClient
from langchain_core.documents import Document

from langchain_foxnose import FoxNoseBatchWriteError, FoxNoseWriter

# Create a FoxNose Flux client
client = FluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)

# Create the writer. external_id_key points at a metadata key holding your own
# stable identifier; its value is sent as the resource key for deduplication
# and is not written into the document's data.
writer = FoxNoseWriter(
    client=client,
    collection_path="knowledge-base",
    page_content_field="body",
    external_id_key="source_id",
)

documents = [
    Document(
        page_content="FoxNose is the knowledge layer for RAG and AI agents.",
        metadata={"title": "What is FoxNose?", "source_id": "docs/what-is-foxnose"},
    ),
    Document(
        page_content="Hybrid search blends keyword and vector relevance.",
        metadata={"title": "Hybrid search", "source_id": "docs/hybrid-search"},
    ),
]

# Every underlying error from a batch write is wrapped in
# FoxNoseBatchWriteError, so branch on exc.cause. A separate
# `except ContentValidationFailed` here would be unreachable.
try:
    keys = writer.add_documents(documents)
except FoxNoseBatchWriteError as exc:
    print(f"Batch failed at document {exc.failed_index} of {exc.total}.")
    print(f"  Written, not rolled back: {exc.written_keys}")
    print(f"  Unknown outcome:          document {exc.failed_index}")
    print(f"  Not attempted:            {exc.pending_indexes}")

    cause = exc.cause
    if isinstance(cause, ContentValidationFailed):
        for error in cause.errors:
            print(f"  Schema error at {error.get('json_path')}: {error}")
    elif isinstance(cause, ExternalIdConflict):
        print("  That source_id already exists in the collection.")
    raise

print(f"Created resources: {keys}\n")

# update_document takes the INTERNAL resource key returned above, and is a
# full-document replace: fields absent from the new document are removed.
revision = writer.update_document(
    keys[0],
    Document(
        page_content="FoxNose is the knowledge layer for RAG and AI agents. Updated.",
        metadata={"title": "What is FoxNose?", "source_id": "docs/what-is-foxnose"},
    ),
)
print(f"New revision: {revision}")

client.close()

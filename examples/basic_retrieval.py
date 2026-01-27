"""Basic retrieval example using FoxNoseRetriever."""

from foxnose_sdk.auth import SimpleKeyAuth
from foxnose_sdk.flux import FluxClient

from langchain_foxnose import FoxNoseRetriever

# Create a FoxNose Flux client
client = FluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)

# Create the retriever with hybrid search (text + vector)
retriever = FoxNoseRetriever(
    client=client,
    folder_path="knowledge-base",
    page_content_field="body",
    search_mode="hybrid",
    top_k=5,
)

# Retrieve documents
docs = retriever.invoke("How do I reset my password?")

for i, doc in enumerate(docs, 1):
    print(f"\n--- Result {i} ---")
    print(f"Key: {doc.metadata.get('key')}")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")

client.close()

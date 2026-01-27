"""Hybrid search example with custom weights and reranking."""

from foxnose_sdk.auth import SimpleKeyAuth
from foxnose_sdk.flux import FluxClient

from langchain_foxnose import FoxNoseRetriever

client = FluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)

# Hybrid search: blends text and vector results with custom weights
retriever = FoxNoseRetriever(
    client=client,
    folder_path="articles",
    page_content_fields=["title", "body"],
    page_content_separator="\n\n",
    search_mode="hybrid",
    top_k=10,
    hybrid_config={
        "vector_weight": 0.7,
        "text_weight": 0.3,
        "rerank_results": True,
    },
    # Only search specific fields
    search_fields=["title", "summary"],
    vector_fields=["body"],
    similarity_threshold=0.65,
)

docs = retriever.invoke("modern approaches to content management")

for doc in docs:
    print(f"[{doc.metadata.get('key')}] {doc.page_content[:100]}...")
    print()

client.close()

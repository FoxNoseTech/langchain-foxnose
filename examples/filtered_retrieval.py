"""Filtered retrieval example using FoxNose structured filters."""

from foxnose_sdk.auth import SimpleKeyAuth
from foxnose_sdk.flux import FluxClient

from langchain_foxnose import FoxNoseRetriever

client = FluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)

# Search with structured filters:
# - Only published articles
# - In tech or science categories
# - Created after January 2024
# - Sorted by newest first
retriever = FoxNoseRetriever(
    client=client,
    folder_path="articles",
    page_content_field="body",
    search_mode="hybrid",
    top_k=10,
    where={
        "$": {
            "all_of": [
                {"status__eq": "published"},
                {
                    "any_of": [
                        {"category__in": ["tech", "science"]},
                        {"tags__includes": "machine-learning"},
                    ]
                },
                {"_sys.created_at__gte": "2024-01-01"},
            ]
        }
    },
    sort=["-_sys.created_at"],
    # Control which metadata fields are returned
    metadata_fields=["title", "category", "tags"],
)

docs = retriever.invoke("machine learning best practices")

for doc in docs:
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Category: {doc.metadata.get('category')}")
    print(f"Content: {doc.page_content[:150]}...")
    print()

client.close()

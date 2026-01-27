"""Async retrieval example using AsyncFluxClient."""

import asyncio

from foxnose_sdk.auth import SimpleKeyAuth
from foxnose_sdk.flux import AsyncFluxClient

from langchain_foxnose import FoxNoseRetriever


async def main() -> None:
    # Create an async FoxNose Flux client
    async_client = AsyncFluxClient(
        base_url="https://<env_key>.fxns.io",
        api_prefix="my_api",
        auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
    )

    # Create the retriever with the async client
    retriever = FoxNoseRetriever(
        async_client=async_client,
        folder_path="knowledge-base",
        page_content_field="body",
        search_mode="hybrid",
        top_k=5,
        similarity_threshold=0.7,
    )

    # Use ainvoke for native async retrieval
    docs = await retriever.ainvoke("How does vector search work?")

    for i, doc in enumerate(docs, 1):
        print(f"\n--- Result {i} ---")
        print(f"Key: {doc.metadata.get('key')}")
        print(f"Content: {doc.page_content[:200]}...")

    # Run multiple searches concurrently
    results = await asyncio.gather(
        retriever.ainvoke("getting started guide"),
        retriever.ainvoke("API authentication"),
        retriever.ainvoke("best practices"),
    )

    for query_docs in results:
        print(f"\nFound {len(query_docs)} documents")

    await async_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())

"""Vector field search examples using custom embeddings.

Demonstrates two approaches:
1. Using a LangChain Embeddings model (converts query text to vector at query time)
2. Using a pre-computed static query vector
"""

from foxnose_sdk.auth import SimpleKeyAuth
from foxnose_sdk.flux import FluxClient

from langchain_foxnose import FoxNoseRetriever

# Create a FoxNose Flux client
client = FluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)


# ---------------------------------------------------------------------------
# Example 1: With a LangChain Embeddings model
# ---------------------------------------------------------------------------
# The retriever calls embeddings.embed_query() on every invocation to convert
# the query text into a vector, then uses vector_field_search() to find
# similar documents.
#
# NOTE: The query text is sent to the embedding provider (e.g. OpenAI).

# from langchain_openai import OpenAIEmbeddings
#
# retriever = FoxNoseRetriever(
#     client=client,
#     folder_path="knowledge-base",
#     page_content_field="body",
#     search_mode="vector",
#     embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
#     vector_field="embedding",        # the field name in FoxNose
#     similarity_threshold=0.75,
#     top_k=5,
# )
#
# docs = retriever.invoke("How do I reset my password?")
# for doc in docs:
#     print(doc.page_content[:100])
#     print(doc.metadata)
#     print("---")


# ---------------------------------------------------------------------------
# Example 2: With a static query vector
# ---------------------------------------------------------------------------
# Useful when you already have an embedding from another source.

retriever = FoxNoseRetriever(
    client=client,
    folder_path="knowledge-base",
    page_content_field="body",
    search_mode="vector",
    query_vector=[0.01, -0.03, 0.15, 0.42, -0.08],  # your pre-computed vector
    vector_field="embedding",
    top_k=5,
)

docs = retriever.invoke("ignored — query_vector is used instead")
for doc in docs:
    print(doc.page_content[:100])
    print(doc.metadata)
    print("---")


# ---------------------------------------------------------------------------
# Example 3: Vector-boosted mode with custom embeddings
# ---------------------------------------------------------------------------
# Text search results are boosted by vector similarity scores.

# from langchain_openai import OpenAIEmbeddings
#
# retriever = FoxNoseRetriever(
#     client=client,
#     folder_path="knowledge-base",
#     page_content_field="body",
#     search_mode="vector_boosted",
#     embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
#     vector_field="embedding",
#     vector_boost_config={
#         "boost_factor": 1.5,
#         "max_boost_results": 20,
#     },
# )
#
# docs = retriever.invoke("password reset procedure")

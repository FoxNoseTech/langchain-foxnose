"""Document loader example — load all documents from a FoxNose folder."""

from foxnose_sdk.auth import SimpleKeyAuth
from foxnose_sdk.flux import FluxClient

from langchain_foxnose import FoxNoseLoader

# Create a FoxNose Flux client
client = FluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)

# Create the loader
loader = FoxNoseLoader(
    client=client,
    folder_path="knowledge-base",
    page_content_field="body",
    params={"where": {"status__eq": "published"}},
    batch_size=50,
)

# Load all documents at once
docs = loader.load()
print(f"Loaded {len(docs)} documents\n")

# Or iterate lazily for large folders
for doc in loader.lazy_load():
    print(f"Key: {doc.metadata.get('key')}")
    print(f"Content: {doc.page_content[:100]}...")
    print()

client.close()

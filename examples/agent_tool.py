"""Agent tool example — use FoxNose search with a LangChain agent."""

from foxnose_sdk.auth import SimpleKeyAuth
from foxnose_sdk.flux import FluxClient

from langchain_foxnose import create_foxnose_tool

# Create a FoxNose Flux client
client = FluxClient(
    base_url="https://<env_key>.fxns.io",
    api_prefix="my_api",
    auth=SimpleKeyAuth("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY"),
)

# Create the search tool
tool = create_foxnose_tool(
    client=client,
    collection_path="knowledge-base",
    page_content_field="body",
    name="kb_search",
    description="Search the knowledge base for relevant information.",
    search_mode="hybrid",
    top_k=5,
)

# Use the tool directly
result = tool.invoke("How do I reset my password?")
print("Search result:")
print(result)

# Use with a LangChain agent (requires langchain and langchain-openai)
#
# from langchain.agents import create_agent
#
# agent = create_agent(model="openai:gpt-4o", tools=[tool])
# response = agent.invoke({
#     "messages": [{"role": "user", "content": "How do I reset my password?"}]
# })
# print(response["messages"][-1].content)

client.close()

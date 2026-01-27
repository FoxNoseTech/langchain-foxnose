"""LangChain integration for FoxNose — the knowledge layer for RAG and AI agents."""

from langchain_foxnose._version import __version__
from langchain_foxnose.retrievers import FoxNoseRetriever

__all__ = ["FoxNoseRetriever", "__version__"]

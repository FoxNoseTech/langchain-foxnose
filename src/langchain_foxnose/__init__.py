"""LangChain integration for FoxNose — the knowledge layer for RAG and AI agents."""

from langchain_foxnose._version import __version__
from langchain_foxnose.loaders import FoxNoseLoader
from langchain_foxnose.retrievers import FoxNoseRetriever
from langchain_foxnose.tools import create_foxnose_tool

__all__ = ["FoxNoseLoader", "FoxNoseRetriever", "__version__", "create_foxnose_tool"]

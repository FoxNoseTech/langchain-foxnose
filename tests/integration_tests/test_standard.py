"""LangChain standard integration tests for FoxNoseRetriever."""

from __future__ import annotations

import os
from typing import Any

from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_foxnose import FoxNoseRetriever


class TestFoxNoseRetrieverStandard(RetrieversIntegrationTests):
    """Standard LangChain retriever integration tests.

    Requires the following environment variables:
        FOXNOSE_BASE_URL: e.g. "https://<env_key>.fxns.io"
        FOXNOSE_API_PREFIX: e.g. "my_api"
        FOXNOSE_PUBLIC_KEY: Flux API public key
        FOXNOSE_SECRET_KEY: Flux API secret key
        FOXNOSE_FOLDER_PATH: Folder containing at least 3 test documents
        FOXNOSE_CONTENT_FIELD: (optional) field name, defaults to "body"
    """

    @property
    def retriever_constructor(self) -> type[FoxNoseRetriever]:
        return FoxNoseRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        from foxnose_sdk.auth import SimpleKeyAuth
        from foxnose_sdk.flux import FluxClient

        client = FluxClient(
            base_url=os.environ["FOXNOSE_BASE_URL"],
            api_prefix=os.environ["FOXNOSE_API_PREFIX"],
            auth=SimpleKeyAuth(
                os.environ["FOXNOSE_PUBLIC_KEY"],
                os.environ["FOXNOSE_SECRET_KEY"],
            ),
        )
        return {
            "client": client,
            "folder_path": os.environ["FOXNOSE_FOLDER_PATH"],
            "page_content_field": os.environ.get("FOXNOSE_CONTENT_FIELD", "body"),
            "search_mode": "vector",
            "similarity_threshold": 0.2,
        }

    @property
    def retriever_query_example(self) -> str:
        return "password reset"

    @property
    def num_results_arg_name(self) -> str:
        return "top_k"

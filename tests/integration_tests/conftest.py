"""Fixtures for the live integration tests.

Every test in this package is skipped unless the read credentials below are
present in the environment. Nothing here has a default that points at a real
environment -- the suite is inert on a fresh checkout, and running it in a clean
shell must skip everything without opening a socket.

The environment must be provisioned to match the contract in the README's
"Running integration tests" section.

NEVER point these tests, or the keys they use, at production or customer data.
Use a synthetic corpus and least-privilege keys: an assertion failure prints the
surrounding document content and metadata into a PUBLIC CI log.

Flux key permissions are scoped to the API PREFIX, not to a collection, so the
write key can create or update any collection under FOXNOSE_API_PREFIX whose
allowed_methods permit it. That prefix must therefore be dedicated to these
synthetic fixtures: the read collection read-only, the throwaway write
collection the only one allowing create/update, and nothing else attached.

FOXNOSE_WRITE_COLLECTION_PATH must be a DEDICATED THROWAWAY collection: Flux has
no delete endpoint, so the writer tests append rows and can never clean up.

Required:
    FOXNOSE_BASE_URL         e.g. "https://<env_key>.fxns.io"
    FOXNOSE_API_PREFIX       Flux API prefix
    FOXNOSE_PUBLIC_KEY       read key, public part
    FOXNOSE_SECRET_KEY       read key, secret part
    FOXNOSE_COLLECTION_PATH  collection holding at least 3 matching documents
    FOXNOSE_QUERY            token matching at least 3 of those documents

Optional (see the README for the full table):
    FOXNOSE_CONTENT_FIELD, FOXNOSE_METADATA_FIELD, FOXNOSE_FILTER_FIELD,
    FOXNOSE_FILTER_VALUE, FOXNOSE_WRITE_PUBLIC_KEY, FOXNOSE_WRITE_SECRET_KEY,
    FOXNOSE_WRITE_COLLECTION_PATH, FOXNOSE_WRITE_CONTENT_FIELD

    FOXNOSE_INTEGRATION_REQUIRED=1 turns every configuration skip below into a
    failure. CI sets it so a release cannot ship with a whole test category
    silently unrun.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any, NoReturn

import pytest

# Each entry is the set of names that satisfy one requirement, so a renamed
# variable can be spelled either way. FOXNOSE_FOLDER_PATH is the pre-0.4.0 name
# for FOXNOSE_COLLECTION_PATH; listing only the new name here would skip the
# whole package before the fallback below ever got a chance to read the old one.
READ_ENV_VARS = [
    ("FOXNOSE_BASE_URL",),
    ("FOXNOSE_API_PREFIX",),
    ("FOXNOSE_PUBLIC_KEY",),
    ("FOXNOSE_SECRET_KEY",),
    ("FOXNOSE_COLLECTION_PATH", "FOXNOSE_FOLDER_PATH"),
    ("FOXNOSE_QUERY",),
]

WRITE_ENV_VARS = [
    "FOXNOSE_WRITE_PUBLIC_KEY",
    "FOXNOSE_WRITE_SECRET_KEY",
    "FOXNOSE_WRITE_COLLECTION_PATH",
]

VECTOR_SKIP_CODES = {
    "vector_search_not_enabled",
    "vector_search_not_available",
    "invalid_request",
    "field_not_found",
}

FLUX_SKIP_CODES = {
    "environment_not_found",
    "api_not_found",
    "folder_not_found",
    "collection_not_found",
    "route_not_found",
    "action_not_allowed",
    "schema_not_available",
}


def skip_unconfigured(reason: str) -> NoReturn:
    """Skip because the environment is not configured for this test.

    Fails instead of skipping when FOXNOSE_INTEGRATION_REQUIRED is set, so the
    main-branch job cannot pass with a test category quietly unrun. Use this for
    MISSING CONFIGURATION only -- for a capability the backend genuinely lacks,
    use skip_if_vector_unavailable / skip_on_flux_unavailable, which always skip.
    """
    if os.environ.get("FOXNOSE_INTEGRATION_REQUIRED"):
        pytest.fail(f"FOXNOSE_INTEGRATION_REQUIRED is set but: {reason}")
    pytest.skip(reason)


def skip_on_flux_unavailable(exc: Any) -> NoReturn:
    """Skip when the backend is not wired for this test, else re-raise."""
    if getattr(exc, "error_code", None) in FLUX_SKIP_CODES:
        pytest.skip(f"Flux API not configured: {exc.error_code}")
    raise exc


def skip_if_vector_unavailable(exc: Any) -> NoReturn:
    """Skip when vector search is not usable here, else re-raise.

    A backend capability, not configuration: this skips even in required mode.
    """
    if getattr(exc, "error_code", None) in VECTOR_SKIP_CODES:
        pytest.skip(f"Vector search unavailable: {exc.error_code}")
    raise exc


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the whole package when the read credentials are absent."""
    missing = [
        " or ".join(names)
        for names in READ_ENV_VARS
        if not any(os.environ.get(name) for name in names)
    ]
    if not missing:
        return
    skip = pytest.mark.skip(reason=f"Missing env vars: {', '.join(missing)}")
    for item in items:
        item.add_marker(skip)


# --- Read-side configuration ---------------------------------------------
#
# Every optional variable falls back with `or`, never with a get() default:
# GitHub Actions exports an UNSET secret as an EMPTY STRING, not as an absent
# key, so os.environ.get("X", "body") yields "" in CI and every lookup silently
# targets a field named "".


@pytest.fixture(scope="session")
def collection_path() -> str:
    """Collection the read tests search. Accepts the pre-0.4.0 variable name."""
    return os.environ.get("FOXNOSE_COLLECTION_PATH") or os.environ["FOXNOSE_FOLDER_PATH"]


@pytest.fixture(scope="session")
def content_field() -> str:
    return os.environ.get("FOXNOSE_CONTENT_FIELD") or "body"


@pytest.fixture(scope="session")
def metadata_field() -> str:
    return os.environ.get("FOXNOSE_METADATA_FIELD") or "title"


@pytest.fixture(scope="session")
def query() -> str:
    return os.environ["FOXNOSE_QUERY"]


@pytest.fixture(scope="session")
def filter_predicate() -> tuple[str, str]:
    """``(field, value)`` for the where-filter test.

    Structured filtering is core retriever functionality, not a backend
    capability, so this uses skip_unconfigured and both variables are in the
    workflow's secrets guard. The collection must hold at least one document
    that matches AND at least one that does not, or the assertions go vacuous.
    """
    field = os.environ.get("FOXNOSE_FILTER_FIELD")
    value = os.environ.get("FOXNOSE_FILTER_VALUE")
    if not field or not value:
        skip_unconfigured(
            "The where-filter test needs FOXNOSE_FILTER_FIELD and FOXNOSE_FILTER_VALUE."
        )
    return field, value


@pytest.fixture(scope="session")
def flux_client() -> Iterator[Any]:
    from foxnose_sdk.auth import SimpleKeyAuth
    from foxnose_sdk.flux import FluxClient

    client = FluxClient(
        base_url=os.environ["FOXNOSE_BASE_URL"],
        api_prefix=os.environ["FOXNOSE_API_PREFIX"],
        auth=SimpleKeyAuth(os.environ["FOXNOSE_PUBLIC_KEY"], os.environ["FOXNOSE_SECRET_KEY"]),
    )
    try:
        yield client
    finally:
        client.close()


@pytest.fixture()
async def async_flux_client() -> Any:
    from foxnose_sdk.auth import SimpleKeyAuth
    from foxnose_sdk.flux import AsyncFluxClient

    client = AsyncFluxClient(
        base_url=os.environ["FOXNOSE_BASE_URL"],
        api_prefix=os.environ["FOXNOSE_API_PREFIX"],
        auth=SimpleKeyAuth(os.environ["FOXNOSE_PUBLIC_KEY"], os.environ["FOXNOSE_SECRET_KEY"]),
    )
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture(scope="session")
def read_corpus_is_sufficient(flux_client: Any, collection_path: str, query: str) -> None:
    """Assert the read collection actually satisfies the fixture contract.

    Without this, a thin collection turns the standard suite's exact-count
    assertions into confusing failures that look like retriever bugs.
    """
    from foxnose_sdk.errors import FoxnoseAPIError

    try:
        response = flux_client.search(
            collection_path,
            body={"search_mode": "text", "find_text": {"query": query}, "limit": 10},
        )
    except FoxnoseAPIError as exc:
        skip_on_flux_unavailable(exc)
    hits = response.get("results", [])
    if len(hits) < 3:
        skip_unconfigured(
            f"FOXNOSE_COLLECTION_PATH must contain at least 3 documents matching "
            f"FOXNOSE_QUERY; a text search returned {len(hits)}."
        )


# --- Write-side configuration --------------------------------------------


@pytest.fixture(scope="session")
def write_collection_path() -> str:
    missing = [name for name in WRITE_ENV_VARS if not os.environ.get(name)]
    if missing:
        skip_unconfigured(f"Writer tests need: {', '.join(missing)}")
    return os.environ["FOXNOSE_WRITE_COLLECTION_PATH"]


@pytest.fixture(scope="session")
def write_content_field(content_field: str) -> str:
    return os.environ.get("FOXNOSE_WRITE_CONTENT_FIELD") or content_field


@pytest.fixture(scope="session")
def write_flux_client(write_collection_path: str) -> Iterator[Any]:
    from foxnose_sdk.auth import SimpleKeyAuth
    from foxnose_sdk.flux import FluxClient

    client = FluxClient(
        base_url=os.environ["FOXNOSE_BASE_URL"],
        api_prefix=os.environ["FOXNOSE_API_PREFIX"],
        auth=SimpleKeyAuth(
            os.environ["FOXNOSE_WRITE_PUBLIC_KEY"],
            os.environ["FOXNOSE_WRITE_SECRET_KEY"],
        ),
    )
    try:
        yield client
    finally:
        client.close()


@pytest.fixture()
async def async_write_flux_client(write_collection_path: str) -> Any:
    from foxnose_sdk.auth import SimpleKeyAuth
    from foxnose_sdk.flux import AsyncFluxClient

    client = AsyncFluxClient(
        base_url=os.environ["FOXNOSE_BASE_URL"],
        api_prefix=os.environ["FOXNOSE_API_PREFIX"],
        auth=SimpleKeyAuth(
            os.environ["FOXNOSE_WRITE_PUBLIC_KEY"],
            os.environ["FOXNOSE_WRITE_SECRET_KEY"],
        ),
    )
    try:
        yield client
    finally:
        await client.aclose()


# --- Objects under test ---------------------------------------------------


@pytest.fixture()
def live_retriever(
    flux_client: Any,
    collection_path: str,
    content_field: str,
    read_corpus_is_sufficient: None,
) -> Any:
    from langchain_foxnose import FoxNoseRetriever

    return FoxNoseRetriever(
        client=flux_client,
        collection_path=collection_path,
        page_content_field=content_field,
        search_mode="text",
        top_k=3,
    )


@pytest.fixture()
def live_writer(
    write_flux_client: Any, write_collection_path: str, write_content_field: str
) -> Any:
    from langchain_foxnose import FoxNoseWriter

    return FoxNoseWriter(
        client=write_flux_client,
        collection_path=write_collection_path,
        page_content_field=write_content_field,
        external_id_key="source_id",
    )

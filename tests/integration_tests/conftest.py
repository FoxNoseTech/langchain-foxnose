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
import pathlib
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

# Codes that mean the BACKEND CANNOT DO THIS AT ALL. Only a genuine missing
# capability belongs here, because these skip even in required mode.
#
# Deliberately absent: "invalid_request" and "field_not_found". Both were here
# once, and both are produced by a request this library built wrongly -- which
# is the exact class of bug these live tests exist to catch. Treating them as
# capability gaps turned a real regression into a green skip.
VECTOR_SKIP_CODES = {
    "vector_search_not_enabled",
    "vector_search_not_available",
}

# Codes that mean the ENVIRONMENT IS NOT SET UP the way the fixture contract
# describes -- a wrong collection path, a prefix without the collection
# attached, a field that is not vectorizable. Configuration, not capability, so
# these route through skip_unconfigured and become failures in required mode.
VECTOR_CONFIG_CODES = {
    "field_not_found",
}

FLUX_CONFIG_CODES = {
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
    """Handle a Flux error that means the environment is not set up, else re-raise.

    Every code here describes wiring the fixture contract requires, so this
    goes through skip_unconfigured: in required mode a misconfigured
    environment must fail rather than quietly retire the test.
    """
    if getattr(exc, "error_code", None) in FLUX_CONFIG_CODES:
        skip_unconfigured(f"Flux API not configured for this test: {exc.error_code}")
    raise exc


def skip_if_vector_unavailable(exc: Any) -> NoReturn:
    """Handle a vector-search error, else re-raise.

    A missing capability always skips -- no fixture can conjure vector support
    the backend does not have. A misconfigured field is the fixture's problem
    and fails in required mode. Anything else is a bug in the request this
    library built, and propagates.
    """
    code = getattr(exc, "error_code", None)
    if code in VECTOR_SKIP_CODES:
        pytest.skip(f"Vector search unavailable: {code}")
    if code in VECTOR_CONFIG_CODES:
        skip_unconfigured(
            f"Vector search rejected the fixture ({code}): the content field must "
            f"be marked vectorizable in the collection schema"
        )
    raise exc


_PACKAGE_DIR = pathlib.Path(__file__).parent


def _belongs_to_this_package(item: pytest.Item) -> bool:
    try:
        path = pathlib.Path(str(item.fspath)).resolve()
    except (AttributeError, OSError):
        return False
    return _PACKAGE_DIR.resolve() in path.parents


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip THIS package when the read credentials are absent.

    pytest calls the hook once for the WHOLE session, including items collected
    from other directories, so the list has to be filtered by path. Without
    that filter a plain `pytest tests/` in a shell with no credentials marked
    the offline unit suite as skipped too, and reported a green run that had
    tested nothing at all.
    """
    mine = [item for item in items if _belongs_to_this_package(item)]
    if not mine:
        return
    missing = [
        " or ".join(names)
        for names in READ_ENV_VARS
        if not any(os.environ.get(name) for name in names)
    ]
    if not missing:
        return

    reason = f"Missing env vars: {', '.join(missing)}"
    if os.environ.get("FOXNOSE_INTEGRATION_REQUIRED"):
        # Required mode must not be satisfiable by supplying no credentials at
        # all: skipping here would let the strict CI job pass having run none
        # of the suite it exists to run.
        raise pytest.UsageError(
            f"FOXNOSE_INTEGRATION_REQUIRED is set, but the integration suite is "
            f"not configured. {reason}"
        )
    skip = pytest.mark.skip(reason=reason)
    for item in mine:
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

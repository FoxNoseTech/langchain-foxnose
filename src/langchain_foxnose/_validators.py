"""Strict validation models for search configuration dicts.

The SDK's ``HybridConfig`` and ``VectorBoostConfig`` models do not reject
unknown keys (they use Pydantic's default ``extra="ignore"``).  These local
strict wrappers add ``extra="forbid"`` so that typos and unsupported keys
are caught early with a clear ``ValidationError``.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class StrictHybridConfig(BaseModel):
    """Strict variant of :class:`~foxnose_sdk.flux.models.HybridConfig`."""

    model_config = ConfigDict(extra="forbid")

    vector_weight: float = 0.6
    text_weight: float = 0.4
    rerank_results: bool = True

    @field_validator("vector_weight", "text_weight")
    @classmethod
    def _weight_range(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("weight must be finite")
        if not 0.0 <= v <= 1.0:
            raise ValueError("weight must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def _weights_sum(self) -> StrictHybridConfig:
        total = self.vector_weight + self.text_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"vector_weight + text_weight must equal 1.0, got {total}")
        return self


class StrictVectorBoostConfig(BaseModel):
    """Strict variant of :class:`~foxnose_sdk.flux.models.VectorBoostConfig`."""

    model_config = ConfigDict(extra="forbid")

    boost_factor: float = 1.5
    similarity_threshold: float | None = None
    max_boost_results: int = 20

    @field_validator("boost_factor")
    @classmethod
    def _boost_positive(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("boost_factor must be finite")
        if v <= 0:
            raise ValueError("boost_factor must be > 0")
        return v

    @field_validator("similarity_threshold")
    @classmethod
    def _threshold_range(cls, v: float | None) -> float | None:
        if v is not None:
            if not math.isfinite(v):
                raise ValueError("similarity_threshold must be finite")
            if not 0.0 <= v <= 1.0:
                raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        return v

    @field_validator("max_boost_results")
    @classmethod
    def _max_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_boost_results must be >= 1")
        return v


# Keys that must never appear in search_kwargs because they conflict
# with SearchRequest fields managed by the retriever.
CONFLICTING_SEARCH_KWARGS = frozenset(
    {
        "search_mode",
        "vector_search",
        "vector_field_search",
        "hybrid_config",
        "vector_boost_config",
        "find_text",
        "find_phrase",
    }
)

# Keys in search_kwargs that map to named SDK method parameters
# rather than being passed through **extra_body.
NAMED_SEARCH_KWARGS = frozenset({"limit", "offset"})


def validate_search_kwargs(search_kwargs: dict[str, Any]) -> None:
    """Raise ``ValueError`` if *search_kwargs* contains conflicting keys."""
    bad = CONFLICTING_SEARCH_KWARGS & search_kwargs.keys()
    if bad:
        raise ValueError(
            f"search_kwargs must not contain keys managed by the retriever: "
            f"{', '.join(sorted(bad))}. Set these via dedicated parameters instead."
        )


def split_search_kwargs(
    search_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split *search_kwargs* into named params and extra_body kwargs.

    Returns:
        A ``(named, extra)`` tuple where *named* contains keys like
        ``limit`` / ``offset`` and *extra* contains the rest.
    """
    named: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for k, v in search_kwargs.items():
        if k in NAMED_SEARCH_KWARGS:
            named[k] = v
        else:
            extra[k] = v
    return named, extra

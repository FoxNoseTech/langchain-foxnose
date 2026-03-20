"""Tests for strict config validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from langchain_foxnose._validators import (
    StrictHybridConfig,
    StrictVectorBoostConfig,
    split_search_kwargs,
    validate_search_kwargs,
)


class TestStrictHybridConfig:
    """StrictHybridConfig validation."""

    def test_valid_defaults(self) -> None:
        cfg = StrictHybridConfig()
        assert cfg.vector_weight == 0.6
        assert cfg.text_weight == 0.4
        assert cfg.rerank_results is True

    def test_rejects_unknown_key(self) -> None:
        with pytest.raises(ValidationError):
            StrictHybridConfig(vector_weight=0.5, text_weight=0.5, typo_key=True)

    def test_rejects_nan_weight(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            StrictHybridConfig(vector_weight=float("nan"), text_weight=0.5)

    def test_rejects_out_of_range_weight(self) -> None:
        with pytest.raises(ValidationError, match=r"between 0\.0 and 1\.0"):
            StrictHybridConfig(vector_weight=1.5, text_weight=-0.5)

    def test_rejects_weights_not_summing_to_one(self) -> None:
        with pytest.raises(ValidationError, match=r"must equal 1\.0"):
            StrictHybridConfig(vector_weight=0.8, text_weight=0.8)


class TestStrictVectorBoostConfig:
    """StrictVectorBoostConfig validation."""

    def test_valid_defaults(self) -> None:
        cfg = StrictVectorBoostConfig()
        assert cfg.boost_factor == 1.5
        assert cfg.similarity_threshold is None
        assert cfg.max_boost_results == 20

    def test_rejects_unknown_key(self) -> None:
        with pytest.raises(ValidationError):
            StrictVectorBoostConfig(boost_factor=1.5, unknown=42)

    def test_rejects_nan_boost_factor(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            StrictVectorBoostConfig(boost_factor=float("nan"))

    def test_rejects_negative_boost_factor(self) -> None:
        with pytest.raises(ValidationError, match="must be > 0"):
            StrictVectorBoostConfig(boost_factor=-1.0)

    def test_rejects_nan_similarity_threshold(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            StrictVectorBoostConfig(similarity_threshold=float("nan"))

    def test_rejects_out_of_range_similarity_threshold(self) -> None:
        with pytest.raises(ValidationError, match=r"between 0\.0 and 1\.0"):
            StrictVectorBoostConfig(similarity_threshold=1.5)

    def test_accepts_valid_similarity_threshold(self) -> None:
        cfg = StrictVectorBoostConfig(similarity_threshold=0.75)
        assert cfg.similarity_threshold == 0.75

    def test_rejects_zero_max_boost_results(self) -> None:
        with pytest.raises(ValidationError, match="must be >= 1"):
            StrictVectorBoostConfig(max_boost_results=0)

    def test_accepts_valid_config(self) -> None:
        cfg = StrictVectorBoostConfig(
            boost_factor=2.0, similarity_threshold=0.8, max_boost_results=10
        )
        assert cfg.boost_factor == 2.0


class TestValidateSearchKwargs:
    """validate_search_kwargs and split_search_kwargs."""

    def test_rejects_conflicting_key(self) -> None:
        with pytest.raises(ValueError, match="must not contain"):
            validate_search_kwargs({"search_mode": "text"})

    def test_accepts_safe_keys(self) -> None:
        validate_search_kwargs({"limit": 50, "where": {}})

    def test_split_extracts_named(self) -> None:
        named, extra = split_search_kwargs({"limit": 50, "offset": 10, "where": {}})
        assert named == {"limit": 50, "offset": 10}
        assert extra == {"where": {}}

    def test_split_empty(self) -> None:
        named, extra = split_search_kwargs({})
        assert named == {}
        assert extra == {}

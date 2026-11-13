"""Configuration models for selecting MCDM weighting and ranking methods."""
from dataclasses import dataclass, field
from typing import Any

from src.mcdm.enums import RankingMethod, WeightingMethod


@dataclass
class WeightingMethodConfig:
    """Method selection and method-specific options for criterion weighting."""

    method: WeightingMethod
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankingMethodConfig:
    """Method selection and method-specific options for alternative ranking."""

    method: RankingMethod
    parameters: dict[str, Any] = field(default_factory=dict)

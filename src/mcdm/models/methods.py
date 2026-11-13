"""Configuration models for selecting MCDM weighting and ranking methods."""
from dataclasses import dataclass, field
from typing import Any

from src.mcdm.enums import RankingMethod, WeightingMethod, MeanMethod, GroupPowerMethod, EmptyGroupPolicy, OnExceedancePolicy


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

@dataclass
class AggregationMethodConfig:
    """Method selection and method-specific options for session aggregation."""
    within_group: MeanMethod
    between_groups: MeanMethod
    group_power_normalization: GroupPowerMethod
    empty_group_policy: EmptyGroupPolicy

@dataclass
class ConsistencyMethodConfig:
    """Method """
    threshold: float
    on_exceedance: OnExceedancePolicy
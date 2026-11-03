from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.enum import RankingMethod, WeightingMethod


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

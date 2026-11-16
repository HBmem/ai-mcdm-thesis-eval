"""Result models for aggregated AHP weights and TOPSIS rankings."""
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

class CriterionWeight(BaseModel):
    """Final AHP weight assigned to one criterion after stakeholder aggregation."""

    criterion_id: str = Field(
        ...,
        description="Identifier of the criterion.",
    )

    criterion_name: str = Field(
        ...,
        description="Human-readable criterion name.",
    )

    weight: float = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "Normalized AHP importance weight assigned to the criterion."
        ),
    )

class WeightResult(BaseModel):
    criterion_weights: list[CriterionWeight] = Field(
        min_length=1,
        description="Calculated weights, identified by criterion.",
    )

    consistency_ratio: float | None = Field(
        default=None,
        ge=0,
        description=(
            "AHP consistency ratio. None when the selected "
            "weighting method does not provide one."
        ),
    )

class AlternativeRanking(BaseModel):
    """TOPSIS ranking result for one policy alternative."""

    alternative_id: str = Field(
        ...,
        description="Identifier of the policy alternative.",
    )

    alternative_name: str = Field(
        ...,
        description="Human-readable name of the policy alternative.",
    )

    rank: int = Field(
        ...,
        ge=1,
        description=(
            "Ordinal TOPSIS rank. Rank 1 represents the highest-ranked "
            "alternative."
        ),
    )

    closeness_coefficient: float = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "TOPSIS closeness coefficient. Higher values indicate that "
            "the alternative is closer to the ideal solution."
        ),
    )

class RankingResult(BaseModel):
    rankings: list[AlternativeRanking] = Field(
        min_length=1,
        description="Alternatives ordered from highest to lowest preference.",
    )

@dataclass
class EntityCalculation:
    entity_id: str
    scope: Literal["participant", "group", "session"]
    pairwise_matrix: list[list[float]]
    weighting: WeightResult
    ranking: RankingResult
    metadata: dict = field(default_factory=dict)


@dataclass
class ExperimentCalculation:
    participants: list[EntityCalculation] = field(default_factory=list)
    groups: list[EntityCalculation] = field(default_factory=list)
    session: EntityCalculation | None = None
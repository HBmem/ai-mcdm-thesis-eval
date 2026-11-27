"""Result models for aggregated AHP weights and TOPSIS rankings."""
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field, computed_field
from typing import Literal

WEIGHT_SUM_TOLERANCE = 1e-10

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

    consistency_threshold: float = Field(
        ge=0,
        description=(
            "Maximum acceptable AHP consistency ratio. If the computed "
            "consistency ratio exceeds this value, the judgment matrix is "
            "considered inconsistent."
        ),
    )
    
    @computed_field
    @property
    def exceeds_consistency_threshold(self) -> bool | None:
        if self.consistency_ratio is None:
            return None

        return (
            self.consistency_ratio
            > self.consistency_threshold
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

StageStatus = Literal[
    "not_run",
    "running",
    "completed",
    "failed",
]

@dataclass
class StageError:
    type: str
    message: str

@dataclass
class StageRecord:
    status: StageStatus = "not_run"
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: StageError | None = None


@dataclass
class EntityStages:
    weighting: StageRecord = field(default_factory=StageRecord)
    ranking: StageRecord = field(default_factory=StageRecord)


@dataclass
class WeightingSource:
    kind: Literal["calculated", "reused"]
    source_run_id: str | None = None
    source_case_key: str | None = None


@dataclass
class EntityMetadata:
    group_id: str | None = None
    participant_count: int | None = None
    voting_power: float | None = None

    weighting_source: WeightingSource | None = None

    ranking_criterion_ids: list[str] = field(
        default_factory=list
    )
    applied_criterion_weights: dict[str, float] = field(
        default_factory=dict
    )
    ranking_weight_source: Literal[
        "original",
        "restricted_and_renormalized",
        "explicit",
    ] | None = None

    stages: EntityStages = field(default_factory=EntityStages)

    # The numerical aggregation layer should record the actual normalized share.
    # None means it was not supplied; the DCL must never infer a value or treat it
    # as zero. Adding this optional slot does not change aggregation calculations.
    effective_voting_power: float | None = None

@dataclass
class EntityCalculation:
    entity_id: str
    scope: Literal["participant", "group", "session"]
    pairwise_matrix: list[list[float]]

    weighting: WeightResult | None = None
    ranking: RankingResult | None = None

    metadata: EntityMetadata = field(default_factory=EntityMetadata)

@dataclass
class ExperimentCalculation:
    participants: list[EntityCalculation] = field(default_factory=list)
    groups: list[EntityCalculation] = field(default_factory=list)
    session: EntityCalculation | None = None

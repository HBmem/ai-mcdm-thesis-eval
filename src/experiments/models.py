import numpy as np

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Literal

from src.mcdm.enums import PairwisePreference
from src.mcdm.models.domain import StakeholderGroup
from src.mcdm.models.methods import RankingMethodConfig, WeightingMethodConfig, AggregationMethodConfig, ConsistencyMethodConfig
from src.mcdm.models.session import DecisionSession
from src.mcdm.models.computation import WeightResult

@dataclass
class ExperimentParticipant:
    participant_id: str
    group_id: str
    judgments: dict[str, PairwisePreference | float]


@dataclass
class ParticipantVariation:
    variant_id: str
    description: str
    participants: list[ExperimentParticipant]


@dataclass
class ExperimentStakeholders:
    stakeholder_groups: list[StakeholderGroup] = field(default_factory=list)
    participants: list[ExperimentParticipant] = field(default_factory=list)
    participant_variants: list[ParticipantVariation] = field(default_factory=list)


@dataclass
class ExperimentResources:
    criteria: str
    alternatives: str
    dataset: str
    dataset_join_key: str | None = None


@dataclass
class ExperimentProcess:
    weighting_method: WeightingMethodConfig
    ranking_method: RankingMethodConfig
    aggregation_method: AggregationMethodConfig
    consistency_method: ConsistencyMethodConfig

@dataclass
class AIConfig:
    enabled: bool

@dataclass
class ExperimentDefaults:
    pipeline: str
    seed: int
    repetition: int
    processing: ExperimentProcess
    ai: AIConfig

@dataclass
class AddedAlternative:
    id: str
    name: str
    values: dict[str, float]

@dataclass
class AddedCriterion:
    id: str
    name: str
    values: dict[str, float]

@dataclass
class ExecutionVariation:
    variant_id: str
    description: str
    parent: str | None = None
    group_judgments: dict[str, dict[str, float]] = field(default_factory=dict)
    participant_judgments: dict[str, dict[str, float]] = field(default_factory=dict)
    participant_counts: dict[str, int] = field(default_factory=dict)

    voting_powers: dict[str, float] = field(default_factory=dict)
    
    add_alternatives: list[AddedAlternative] = field(default_factory=list)
    add_criteria: list[AddedCriterion] = field(default_factory=list)
    remove_winner: bool = False
    remove_criteria: list[str] = field(default_factory=list)
    remove_alternatives: list[str] = field(default_factory=list)

    weights: dict[str, float] | None = None
    reuse_parent_weights: bool = False
    inherit_parent_alternatives: bool = False
    target_profiles: dict[str, list[float]] = field(default_factory=dict)
    sensitivity_criterion: str | None = None
    sensitivity_weight: float | None = None
    compare_to: str | None = None
    expected_assertions: list[dict] = field(default_factory=list)

@dataclass
class ExperimentExecution:
    random_seed: int
    variants: bool = False
    variations: list[ExecutionVariation] = field(default_factory=list)
    consistency_threshold: float = 0.10

@dataclass
class ExperimentAI:
    enabled: bool = False

@dataclass
class Experiment:
    """Loaded inputs for one executable run; construction performs no calculations."""

    schema_version: str
    id: str
    name: str
    research_description: str
    research_purpose: str
    scenario_id: str
    scenario_description: str
    scenario_policy_question: str
    resources: ExperimentResources
    processing: ExperimentProcess
    execution: ExperimentExecution
    ai: ExperimentAI
    session: DecisionSession
    source_path: str = ""
    variant_id: str | None = None
    variant_description: str | None = None
    stakeholders: ExperimentStakeholders = field(default_factory=ExperimentStakeholders)
    expected_assertions: list[dict[str, Any]] = field(default_factory=list)
    ai_evaluation: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    defaults: ExperimentDefaults | None = None
    case_metadata: dict[str, Any] = field(default_factory=dict)
    baseline_case: str | None = None
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    judgment_encoding: str = "pairwise_preference"
    scale_conversion: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON-compatible data, including the domain session."""
        return _to_plain(self)

def _to_plain(value: Any) -> Any:
    if isinstance(value, (DecisionSession, StakeholderGroup)):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _to_plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value

@dataclass
class ReusedWeighting:
    weighting: WeightResult
    pairwise_matrix: np.ndarray
    source_run_id: str
    source_case_key: str


@dataclass
class ResolvedExecutionInputs:
    experiment: Experiment
    reused_weightings: dict[tuple[str, str], ReusedWeighting] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    # Ranking dimensions and interventions are separate from original AHP evidence.
    ranking_criterion_ids: list[str] = field(default_factory=list)
    criterion_weight_overrides: dict[str, dict[str, float]] = field(default_factory=dict)

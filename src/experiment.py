from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any

from src.enum import PairwisePreference
from src.variations import ExecutionVariation
from src.models.domain import StakeholderGroup
from src.models.methods import RankingMethodConfig, WeightingMethodConfig
from src.models.session import DecisionSession


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


@dataclass
class ExperimentExecution:
    random_seed: int
    variants: bool = False
    variations: list[ExecutionVariation] = field(default_factory=list)
    consistency_threshold: float = 0.10


@dataclass
class Experiment:
    """Loaded inputs for one executable run; construction performs no calculations."""

    schema_version: str
    id: str
    name: str
    research_purpose: str
    scenario_id: str
    scenario_description: str
    scenario_policy_question: str
    resources: ExperimentResources
    processing: ExperimentProcess
    execution: ExperimentExecution
    session: DecisionSession
    source_path: str = ""
    variant_id: str | None = None
    variant_description: str | None = None
    stakeholders: ExperimentStakeholders = field(default_factory=ExperimentStakeholders)
    expected_assertions: list[dict[str, Any]] = field(default_factory=list)
    ai_evaluation: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON-compatible data, including the domain session."""
        return _to_plain(self)


def _to_plain(value: Any) -> Any:
    if isinstance(value, (DecisionSession, StakeholderGroup, ExecutionVariation)):
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

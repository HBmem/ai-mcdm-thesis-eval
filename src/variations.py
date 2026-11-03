"""Strict, declarative operations applied independently to base experiment inputs."""
from pydantic import BaseModel, ConfigDict, Field


class AddedAlternative(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    values: dict[str, float]


class ExecutionVariation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    variant_id: str
    description: str
    parent: str | None = None
    group_judgments: dict[str, dict[str, float]] = Field(default_factory=dict)
    participant_judgments: dict[str, dict[str, float]] = Field(default_factory=dict)
    participant_counts: dict[str, int] = Field(default_factory=dict)
    voting_powers: dict[str, float] = Field(default_factory=dict)
    remove_alternatives: list[str] = Field(default_factory=list)
    add_alternatives: list[AddedAlternative] = Field(default_factory=list)
    remove_winner: bool = False
    remove_criteria: list[str] = Field(default_factory=list)
    weights: dict[str, float] | None = None
    reuse_parent_weights: bool = False
    inherit_parent_alternatives: bool = False
    target_profiles: dict[str, list[float]] = Field(default_factory=dict)
    sensitivity_criterion: str | None = None
    sensitivity_weight: float | None = None
    compare_to: str | None = None
    expected_assertions: list[dict] = Field(default_factory=list)

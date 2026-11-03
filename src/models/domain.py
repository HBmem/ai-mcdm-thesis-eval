"""

"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.enum import CriterionDirection, PairwisePreference

class Criterion(BaseModel):
    """
    A factor used to evaluate and rank policy alternatives.

    Criteria represent the dimensions that matter in the decision,
    such as implementation cost, crime reduction, community trust,
    or implementation time.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description=(
            "Stable machine-readable identifier for the criterion. "
            "Example: 'community_trust_impact'."
        ),
        examples=["community_trust_impact"],
    )

    name: str = Field(
        ...,
        description=(
            "Human-readable name of the criterion shown to participants "
            "and AI systems."
        ),
        examples=["Community Trust Impact"],
    )

    description: str = Field(
        ...,
        description=(
            "Explanation of what this criterion measures and why it matters "
            "when comparing policy alternatives."
        ),
    )

    direction: CriterionDirection = Field(
        ...,
        description=(
            "Optimization direction for the criterion. "
            "'max' means larger values are preferred; "
            "'min' means smaller values are preferred."
        ),
    )

    unit: str | None = Field(
        default=None,
        description=(
            "Optional unit used to measure the criterion, such as dollars, "
            "months, percentage points, or an index score."
        ),
        examples=["USD", "months", "percentage"],
    )

class CriterionValue(BaseModel):
    """
    Performance of one policy alternative on one evaluation criterion.
    """

    model_config = ConfigDict(extra="forbid")

    criterion_id: str = Field(
        ...,
        description="Identifier of the criterion being measured.",
    )

    value: float = Field(
        ...,
        description=(
            "Numeric performance value of the alternative for this criterion."
        ),
    )

class Alternative(BaseModel):
    """
    A policy option being evaluated by the MCDM process.

    Each alternative contains quantitative performance values for
    every criterion used by TOPSIS.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Stable machine-readable identifier for the alternative.",
    )

    name: str = Field(
        ...,
        description="Human-readable name of the policy alternative.",
    )

    description: str = Field(
        ...,
        description=(
            "Summary of the policy alternative, including what intervention "
            "or action it represents."
        ),
    )

    criterion_values: list[CriterionValue] = Field(
        ...,
        description=(
            "Measured or estimated performance of this alternative for each "
            "decision criterion."
        ),
    )

    @model_validator(mode="after")
    def criterion_values_must_be_unique(self) -> "Alternative":
        ids = [
            criterion_value.criterion_id
            for criterion_value in self.criterion_values
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "An alternative cannot contain multiple values "
                "for the same criterion."
            )

        return self

class Participant(BaseModel):
    """
    A stakeholder who submitted criterion preferences for an MCDM session.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Unique identifier for the participant.",
    )

    name: str = Field(
        ...,
        description="Display name of the participant.",
        examples=["Maya Thompson"],
    )

    stakeholder_group_id: str = Field(
        ...,
        description=(
            "Identifier of the stakeholder group represented by this "
            "participant."
        ),
        examples=["community_residents"],
    )

    responses: list[PairwiseResponse] = Field(
        ...,
        min_length=1,
        description=(
            "Pairwise criterion comparisons submitted by this participant. "
            "Together these responses are used to construct the participant's "
            "AHP comparison matrix."
        ),
    )

    @model_validator(mode="after")
    def response_questions_must_be_unique(self) -> "Participant":
        ids = [response.question_id for response in self.responses]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "A participant cannot provide multiple responses "
                "to the same question."
            )

        return self

class StakeholderGroup(BaseModel):
    """
    A stakeholder constituency participating in the policy evaluation.

    Stakeholder groups are used during two-stage aggregation so that
    participation volume alone does not determine a group's influence.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description=(
            "Stable machine-readable identifier for the stakeholder group."
        ),
        examples=["community_residents"],
    )

    name: str = Field(
        ...,
        description="Human-readable name of the stakeholder group.",
        examples=["Community Residents"],
    )

    description: str = Field(
        ...,
        description=(
            "Description of the constituency represented by this stakeholder "
            "group and its perspective in the policy decision."
        ),
    )

    voting_power: float = Field(
        ...,
        gt=0,
        le=1,
        description=(
            "Relative voting power assigned to this stakeholder group during "
            "between-group aggregation. Active group weights are normalized "
            "before aggregation."
        ),
    )

class PairwiseQuestion(BaseModel):
    """
    A question asking a participant to compare the relative importance
    of two decision criteria.

    The participant's response determines one element of the AHP
    pairwise comparison matrix.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Stable identifier for the pairwise comparison question.",
        examples=["q1"],
    )

    text: str = Field(
        ...,
        description="Question displayed to the participant.",
    )

    left_criterion_id: str = Field(
        ...,
        description=(
            "Identifier of the criterion shown on the left side of the "
            "pairwise comparison."
        ),
    )

    right_criterion_id: str = Field(
        ...,
        description=(
            "Identifier of the criterion shown on the right side of the "
            "pairwise comparison."
        ),
    )

    @model_validator(mode="after")
    def criteria_must_differ(self) -> "PairwiseQuestion":
        if self.left_criterion_id == self.right_criterion_id:
            raise ValueError(
                "A pairwise question must compare two different criteria."
            )

        return self

class PairwiseResponse(BaseModel):
    """
    One participant's answer to one pairwise comparison question.

    The preference indicates whether the left or right criterion is
    considered more important and by approximately how much.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(
        ...,
        description=(
            "Identifier of the pairwise question being answered."
        ),
        examples=["q1"],
    )

    preference: PairwisePreference | float = Field(
        ...,
        description=(
            "Participant's relative preference between the two criteria. "
            "The enum name explains the direction of preference while the "
            "stored value preserves compatibility with legacy pairwise data."
        ),
    )
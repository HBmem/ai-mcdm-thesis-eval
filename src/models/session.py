from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.domain import Criterion, StakeholderGroup, Participant, Alternative, PairwiseQuestion
class DecisionSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Unique identifier for the decision-making session.",
    )

    title: str = Field(
        ...,
        description="Human-readable title of the policy evaluation.",
    )

    description: str = Field(
        ...,
        description=(
            "Explanation of the decision problem being evaluated and "
            "the purpose of the session."
        ),
    )

    criteria: list[Criterion] = Field(
        ...,
        min_length=2,
        description=(
            "Ordered criteria used by AHP and TOPSIS. "
            "The ordering is preserved when matrices are generated."
        ),
    )

    stakeholder_groups: list[StakeholderGroup] = Field(
        ...,
        min_length=1,
        description=(
            "Stakeholder constituencies represented in the decision process."
        ),
    )

    questions: list[PairwiseQuestion] = Field(
        ...,
        min_length=1,
        description=(
            "Pairwise criterion-comparison questions used to collect "
            "AHP preferences."
        ),
    )

    participants: list[Participant] = Field(
        ...,
        min_length=1,
        description="Participants who submitted preferences for the session.",
    )

    alternatives: list[Alternative] = Field(
        default_factory=list,
        description=(
            "Policy alternatives to evaluate using the aggregated criterion "
            "weights and TOPSIS."
        ),
    )

    @model_validator(mode="after")
    def validate_references(self) -> "DecisionSession":

        criterion_ids = {criterion.id for criterion in self.criteria}
        stakeholder_ids = {
            group.id for group in self.stakeholder_groups
        }
        question_ids = {question.id for question in self.questions}

        # Validate question criteria
        for question in self.questions:
            if question.left_criterion_id not in criterion_ids:
                raise ValueError(
                    f"Question '{question.id}' references unknown criterion "
                    f"'{question.left_criterion_id}'."
                )

            if question.right_criterion_id not in criterion_ids:
                raise ValueError(
                    f"Question '{question.id}' references unknown criterion "
                    f"'{question.right_criterion_id}'."
                )

        # Validate participant references
        for participant in self.participants:
            if participant.stakeholder_group_id not in stakeholder_ids:
                raise ValueError(
                    f"Participant '{participant.name}' references unknown "
                    f"stakeholder group "
                    f"'{participant.stakeholder_group_id}'."
                )

            response_ids = {
                response.question_id
                for response in participant.responses
            }

            unknown_questions = response_ids - question_ids

            if unknown_questions:
                raise ValueError(
                    f"Participant '{participant.name}' contains responses "
                    f"for unknown questions: {unknown_questions}"
                )

        return self
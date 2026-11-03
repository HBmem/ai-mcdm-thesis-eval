from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

class CriterionWeight(BaseModel):
    """
    Final AHP weight assigned to one criterion after stakeholder aggregation.
    """

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

class AlternativeRanking(BaseModel):
    """
    TOPSIS ranking result for one policy alternative.
    """

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

class MCDMResult(BaseModel):
    """
    Deterministic result of the AHP-TOPSIS evaluation.

    This object should generally be given to the AI explanation layer rather
    than asking the AI to calculate rankings itself.
    """

    criterion_weights: list[CriterionWeight] = Field(
        ...,
        description="AHP weights generated for each evaluation criterion.",
    )

    consistency_ratio: float = Field(
        ...,
        description=(
            "AHP consistency ratio for the aggregated pairwise comparison "
            "matrix. Lower values indicate more internally consistent "
            "comparisons."
        ),
    )

    rankings: list[AlternativeRanking] = Field(
        ...,
        description=(
            "TOPSIS ranking of all evaluated alternatives from highest "
            "to lowest preference."
        ),
    )
import numpy as np

from dataclasses import dataclass, field
from typing import Literal

from src.mcdm.models.computation import ExperimentCalculation
from src.mcdm.service import validate_weight_vector


RunStatus = Literal[
    "passed",
    "failed",
    "blocked",
    "calculated",
]

def validate_entity_numerics(
    calculation,
    *,
    session,
    processing,
):
    criterion_ids = [
        item["id"]
        for item in session["criteria"]
    ]
    alternative_ids = [
        item["id"]
        for item in session["alternatives"]
    ]

    if (
        not criterion_ids
        or len(criterion_ids) != len(set(criterion_ids))
    ):
        raise ValueError("Invalid input criterion IDs.")

    if (
        not alternative_ids
        or len(alternative_ids) != len(set(alternative_ids))
    ):
        raise ValueError("Invalid input alternative IDs.")

    if calculation.weighting is None:
        raise ValueError("Missing entity weighting.")

    if calculation.ranking is None:
        raise ValueError("Missing entity ranking.")

    count = len(criterion_ids)

    matrix = np.asarray(
        calculation.pairwise_matrix,
        dtype=float,
    )

    if (
        matrix.shape != (count, count)
        or not np.all(np.isfinite(matrix))
        or np.any(matrix <= 0)
    ):
        raise ValueError("Invalid stored pairwise matrix.")

    if not np.allclose(
        np.diag(matrix),
        1.0,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Stored matrix has an invalid diagonal.")

    if not np.allclose(
        np.log(matrix) + np.log(matrix.T),
        0.0,
        atol=1e-9,
        rtol=0,
    ):
        raise ValueError("Stored matrix is not reciprocal.")

    weighting = calculation.weighting

    weights_by_id = {
        item.criterion_id: item.weight
        for item in weighting.criterion_weights
    }

    if (
        len(weights_by_id) != len(weighting.criterion_weights)
        or set(weights_by_id) != set(criterion_ids)
    ):
        raise ValueError("Stored weights must cover the full criteria exactly once.")

    is_ahp = (
        processing["weighting_method"]["method"] == "ahp"
    )

    validate_weight_vector(
        [weights_by_id[cid] for cid in criterion_ids],
        expected_count=count,
        label="Stored entity weights",
        allow_zero=not is_ahp,
    )

    ratio = weighting.consistency_ratio

    if ratio is None:
        if is_ahp:
            raise ValueError("AHP results require a consistency ratio.")

    elif not np.isfinite(ratio) or ratio < 0:
        raise ValueError("Invalid stored consistency ratio.")

    threshold = weighting.consistency_threshold
    expected_threshold = processing["consistency_method"]["threshold"]

    if (
        not np.isfinite(threshold)
        or threshold < 0
        or threshold != expected_threshold
    ):
        raise ValueError("Incorrect stored consistency threshold.")

    selected = calculation.metadata.ranking_criterion_ids

    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(cid, str) for cid in selected)
        or len(selected) != len(set(selected))
        or not set(selected) <= set(criterion_ids)
    ):
        raise ValueError("Invalid ranking criterion IDs.")

    applied = calculation.metadata.applied_criterion_weights

    if (
        not isinstance(applied, dict)
        or set(applied) != set(selected)
    ):
        raise ValueError("Applied weights do not cover the ranking criteria.")

    applied_vector = validate_weight_vector(
        [applied[cid] for cid in selected],
        expected_count=len(selected),
        label="Stored applied ranking weights",
    )

    source = calculation.metadata.ranking_weight_source

    if source not in {
        "original",
        "restricted_and_renormalized",
        "explicit",
    }:
        raise ValueError("Unknown ranking-weight source.")

    if source != "explicit":
        expected = np.asarray([
            weights_by_id[cid]
            for cid in selected
        ])

        restricted = set(selected) != set(criterion_ids)

        if source == "original" and restricted:
            raise ValueError("Original weights cannot omit criteria.")

        if source == "restricted_and_renormalized":
            if not restricted or expected.sum() <= 0:
                raise ValueError("Invalid weight restriction.")

            expected = expected / expected.sum()

        if not np.allclose(
            applied_vector,
            expected,
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError("Applied weights disagree with their declared source.")

    rows = calculation.ranking.rankings
    ranked_ids = [item.alternative_id for item in rows]

    if (
        len(ranked_ids) != len(set(ranked_ids))
        or set(ranked_ids) != set(alternative_ids)
    ):
        raise ValueError("Rankings must cover resolved alternatives exactly once.")

    ranks = [item.rank for item in rows]

    if (
        any(type(rank) is not int for rank in ranks)
        or ranks != list(range(1, len(rows) + 1))
    ):
        raise ValueError("Stored ordinal ranks must be consecutive and ordered.")

    scores = np.asarray([
        item.closeness_coefficient
        for item in rows
    ])

    if (
        not np.all(np.isfinite(scores))
        or np.any(scores < 0)
        or np.any(scores > 1)
    ):
        raise ValueError("Invalid stored ranking scores.")

    if np.any(np.diff(scores) > 1e-12):
        raise ValueError("Ranking scores are not ordered highest to lowest.")

@dataclass
class RunResult:
    experiment_id: str
    case_key: str
    name: str
    run_id: str
    status: RunStatus

    schema_version: Literal["2.0"] = "2.0"
    variant_id: str | None = None
    parent_run_id: str | None = None

    provenance: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    comparison: dict | None = None
    assertions: list[dict] = field(default_factory=list)
    execution: dict = field(default_factory=dict)

    mcdm: ExperimentCalculation = field(
        default_factory=ExperimentCalculation
    )
    # comprehension: ComprehensionResult | None = None
    
    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "case_key",
            "name",
            "run_id",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{field_name} must be a nonempty string."
                )

        if self.schema_version != "2.0":
            raise ValueError("Unsupported result schema version.")

        if not isinstance(self.mcdm, ExperimentCalculation):
            raise TypeError(
                "mcdm must be an ExperimentCalculation."
            )

        self.validate_for_status(self.status)

    def validate_for_status(self, status: RunStatus) -> None:
        if status not in {
            "passed",
            "failed",
            "blocked",
            "calculated",
        }:
            raise ValueError(f"Invalid run status: {status!r}")

        if status in {"calculated", "passed"}:
            self.validate_numerical_complete()

        if status == "passed":
            if self.execution.get("validation_status") != "passed":
                raise ValueError(
                    "Passed results require successful "
                    "evaluation of all required checks."
                )

    def set_status(self, status: RunStatus) -> None:
        # Validate first: a failure leaves the existing status intact.
        self.validate_for_status(status)
        self.status = status

    def validate_numerical_complete(self) -> None:
        session = self.inputs.get("session")

        if not isinstance(session, dict):
            raise ValueError(
                "Complete results require the input session snapshot."
            )

        expected_participants = {
            participant["id"]
            for participant in session["participants"]
        }

        expected_groups = {
            participant["stakeholder_group_id"]
            for participant in session["participants"]
        }

        def check_entities(calculations, expected_ids, scope):
            actual_ids = [
                calculation.entity_id
                for calculation in calculations
            ]

            if len(actual_ids) != len(set(actual_ids)):
                raise ValueError(
                    f"Duplicate {scope} calculation IDs."
                )

            if set(actual_ids) != expected_ids:
                raise ValueError(
                    f"Incomplete or unexpected {scope} calculations."
                )

            if any(
                calculation.scope != scope
                for calculation in calculations
            ):
                raise ValueError(
                    f"Incorrect scope in {scope} calculations."
                )

        check_entities(
            self.mcdm.participants,
            expected_participants,
            "participant",
        )

        check_entities(
            self.mcdm.groups,
            expected_groups,
            "group",
        )

        if self.mcdm.session is None:
            raise ValueError("Missing session calculation.")

        check_entities(
            [self.mcdm.session],
            {session["id"]},
            "session",
        )

        calculations = [
            *self.mcdm.participants,
            *self.mcdm.groups,
            self.mcdm.session,
        ]

        for calculation in calculations:
            validate_entity_numerics(
                calculation,
                session=session,
                processing=self.inputs["processing"],
            )

            stages = calculation.metadata.stages

            if (
                stages.weighting.status != "completed"
                or stages.ranking.status != "completed"
            ):
                raise ValueError(
                    "Complete entity results require completed "
                    "weighting and ranking stages."
                )
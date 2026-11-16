from dataclasses import dataclass, field
from typing import Literal

from src.mcdm.models.computation import ExperimentCalculation



RunStatus = Literal[
    "passed",
    "failed",
    "blocked",
    "calculated",
]


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
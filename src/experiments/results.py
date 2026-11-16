from dataclasses import dataclass, field
from typing import Literal

from src.mcdm.models.computation import ExperimentCalculation

@dataclass
class RunResult:
    experiment_id: str
    case_key: str
    name: str
    run_id: str
    status: Literal["passed", "failed", "blocked", "calculated"]

    # Schema 1.0 for no ai and 1.1 for ai
    schema_version: Literal["1.0", "1.1"] = "1.0"
    variant_id: str | None = None
    parent_run_id: str | None = None

    provenance: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    preferences: dict = field(default_factory=dict)
    applied_weights: dict = field(default_factory=dict)
    topsis: dict = field(default_factory=dict)
    comparison: dict | None = None
    assertions: list[dict] = field(default_factory=list)
    execution: dict = field(default_factory=dict)

    mcdm: ExperimentCalculation = field(
        default_factory=ExperimentCalculation
    )
    # comprehension: ComprehensionResult | None = None
    
    def __post_init__(self):
        # if self.schema_version == "1.1" and self.status == "passed" and (self.comprehension is None or self.comprehension.status != "passed"):
        #     raise ValueError("Passed 1.1 results require successful comprehension.")
        if self.status == "passed" and not (self.inputs and self.preferences and self.topsis and self.applied_weights):
            raise ValueError("Passed results require inputs and a complete scientific trace.")
        if self.preferences:
            for key in ("criterion_ids", "participants", "groups", "session", "source"):
                if key not in self.preferences:
                    raise ValueError(f"Missing preference trace section: {key}")
            for item in [*self.preferences["participants"], *self.preferences["groups"], self.preferences["session"]]:
                required = {"id", "matrix", "weights", "wd", "consistency_ratio", "flagged", "threshold", "lambda_max", "consistency_convention"}
                if not required <= item.keys():
                    raise ValueError("Incomplete preference calculation.")
                matrix = item["matrix"]
                if not {"row_ids", "column_ids", "values"} <= matrix.keys():
                    raise ValueError("Matrices require ordered row and column IDs.")
        if self.topsis:
            required = {"rankings", "decision_matrix", "normalized_matrix", "weighted_matrix", "ideal_best", "ideal_worst"}
            if not required <= self.topsis.keys() or not self.topsis["rankings"]:
                raise ValueError("Incomplete TOPSIS trace.")
        return self
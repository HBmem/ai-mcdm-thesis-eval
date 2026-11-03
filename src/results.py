"""Versioned artifact contract and atomic batch persistence."""
import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    experiment_id: str
    variant_id: str | None = None
    case_key: str
    name: str
    run_id: str
    parent_run_id: str | None = None
    status: Literal["passed", "failed", "blocked"]
    provenance: dict = Field(default_factory=dict)
    inputs: dict = Field(default_factory=dict)
    preferences: dict = Field(default_factory=dict)
    applied_weights: dict = Field(default_factory=dict)
    topsis: dict = Field(default_factory=dict)
    comparison: dict | None = None
    assertions: list[dict] = Field(default_factory=list)
    execution: dict = Field(default_factory=dict)


    @model_validator(mode="after")
    def validate_trace(self):
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


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, payload: str):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def csv_text(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


class BatchStore:
    def __init__(self, root):
        self.batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
        self.path = Path(root).resolve() / self.batch_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.manifest = {"schema_version": "1.0", "batch_id": self.batch_id, "created_at": now(), "status": "running", "runs": []}
        self._manifest()

    def _manifest(self):
        atomic_write(self.path / "manifest.json", json.dumps(self.manifest, indent=2, allow_nan=False))

    def save(self, result: RunResult):
        # Filenames depend on generated UUIDs, never user-provided case identifiers.
        if self.manifest["status"] != "running" or any(e["run_id"] == result.run_id for e in self.manifest["runs"]):
            raise ValueError("Saved runs and completed batches are immutable.")
        result = RunResult.model_validate(result.model_dump())
        filename = result.run_id + ".json"
        atomic_write(self.path / filename, result.model_dump_json(indent=2))
        self.manifest["runs"].append({"case_key": result.case_key, "experiment_id": result.experiment_id,
                                      "variant_id": result.variant_id, "run_id": result.run_id,
                                      "status": result.status, "file": filename})
        self._manifest()

    def finalize(self, results):
        rows, rankings, weights, checks = [], [], [], []
        for r in results:
            rows.append({"case_key": r.case_key, "run_id": r.run_id, "experiment_id": r.experiment_id, "status": r.status,
                         "winner": r.topsis.get("rankings", [{}])[0].get("alternative_id", ""),
                         "session_cr": r.preferences.get("session", {}).get("consistency_ratio", "")})
            rankings.extend({"case_key": r.case_key, **x} for x in r.topsis.get("rankings", []))
            weights.extend({"case_key": r.case_key, "criterion_id": k, "weight": v} for k, v in r.applied_weights.items())
            checks.extend({"case_key": r.case_key, "id": x["id"], "check": x["check"], "status": x["status"]} for x in r.assertions)
        for filename, data, columns in (
            ("summary.csv", rows, ["case_key", "run_id", "experiment_id", "status", "winner", "session_cr"]),
            ("rankings.csv", rankings, ["case_key", "alternative_id", "score", "rank", "statistical_rank", "distance_best", "distance_worst"]),
            ("weights.csv", weights, ["case_key", "criterion_id", "weight"]),
            ("assertions.csv", checks, ["case_key", "id", "check", "status"]),
        ):
            atomic_write(self.path / filename, csv_text(data, columns))
        self.manifest.update(status="completed", completed_at=now(), successful=all(r.status == "passed" for r in results))
        self._manifest()


def safe_artifact(batch: Path, filename: str):
    path = (batch / filename).resolve()
    if path.parent != batch.resolve() or path.suffix != ".json":
        raise ValueError("Manifest artifact must be a JSON file inside its batch.")
    return path

import json
import os
import numpy as np
import re

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from uuid import uuid4
from pydantic import BaseModel


OUTPUT_FLAG_DEFAULTS = {
    "save_resolved_inputs": True,
    "save_intermediate_matrices": True,
    "save_per_participant_results": True,
    "save_per_group_results": True,
    "save_provenance": True,
    "save_assertion_results": True,
    "save_comparisons": True,
    "save_ai_calls": False,
    "save_ai_responses": False,
}


def output_flags(settings):
    flags = {}

    for name, default in OUTPUT_FLAG_DEFAULTS.items():
        value = settings.get(name, default)

        if not isinstance(value, bool):
            raise ValueError(
                f"output.{name} must be a boolean."
            )

        flags[name] = value

    return flags

class ArtifactWriteError(RuntimeError):
    pass

def to_jsonable(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: to_jsonable(getattr(value, item.name))
            for item in fields(value)
        }

    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="python"))

    if isinstance(value, Enum):
        return to_jsonable(value.value)

    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())

    if isinstance(value, np.generic):
        return to_jsonable(value.item())

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings.")

        return {
            key: to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    raise TypeError(
        f"Unsupported artifact value: {type(value).__name__}"
    )

def write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(
        f".{path.name}.{uuid4().hex}.tmp"
    )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                to_jsonable(value),
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)

    except (OSError, TypeError, ValueError) as error:
        raise ArtifactWriteError(
            f"Could not write {path}: {error}"
        ) from error

    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass

def directory_component(value):
    if (
        not isinstance(value, str)
        or value in {"", ".", ".."}
        or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None
    ):
        raise ValueError(
            f"Unsupported artifact directory component: {value!r}"
        )

    return value

def save_run(batch_directory, run, result):
    directory = (
        Path(batch_directory)
        / "experiments"
        / directory_component(result.experiment_id)
        / directory_component(
            run.experiment.case_metadata["case_id"]
        )
        / f"r{run.repetition:03d}"
        / directory_component(result.run_id)
    )

    flags = output_flags(run.experiment.output)

    if flags["save_resolved_inputs"]:
        write_json_atomic(
            directory / "inputs/requested.json",
            run.experiment.to_dict(),
        )
        write_json_atomic(
            directory / "inputs/resolved.json",
            result.inputs,
        )

    write_json_atomic(
        directory / "metadata/execution.json",
        result.execution,
    )

    if flags["save_provenance"]:
        write_json_atomic(
            directory / "metadata/provenance.json",
            result.provenance,
        )
    # write_json_atomic(
    #     directory / "inputs/requested.json",
    #     run.experiment.to_dict(),
    # )
    # write_json_atomic(
    #     directory / "inputs/resolved.json",
    #     result.inputs,
    # )
    # write_json_atomic(
    #     directory / "metadata/execution.json",
    #     result.execution,
    # )
    # write_json_atomic(
    #     directory / "metadata/provenance.json",
    #     result.provenance,
    # )

    def save_entity(path, calculation):
        if flags["save_intermediate_matrices"]:
            write_json_atomic(
                path / "pairwise_matrix.json",
                calculation.pairwise_matrix,
            )

        write_json_atomic(
            path / "weighting.json",
            calculation.weighting,
        )
        write_json_atomic(
            path / "ranking.json",
            calculation.ranking,
        )
        write_json_atomic(
            path / "metadata.json",
            calculation.metadata,
        )

    if flags["save_per_participant_results"]:
        for calculation in result.mcdm.participants:
            save_entity(
                directory
                / "mcdm/participants"
                / directory_component(calculation.entity_id),
                calculation,
            )

    if flags["save_per_group_results"]:
        for calculation in result.mcdm.groups:
            save_entity(
                directory
                / "mcdm/groups"
                / directory_component(calculation.entity_id),
                calculation,
            )

    if result.mcdm.session is not None:
        save_entity(
            directory / "mcdm/session",
            result.mcdm.session,
        )

    if flags["save_assertion_results"]:
        write_json_atomic(
            directory / "assertions/results.json",
            result.assertions,
        )

    if flags["save_comparisons"]:
        write_json_atomic(
            directory / "comparisons/results.json",
            result.comparison,
        )

    # Write the authoritative snapshot after its derived files.
    write_json_atomic(directory / "result.json", result)

    return {
        "execution_key": run.key,
        "case_key": result.case_key,
        "run_id": result.run_id,
        "status": result.status,
        "directory": directory.relative_to(
            batch_directory
        ).as_posix(),
    }

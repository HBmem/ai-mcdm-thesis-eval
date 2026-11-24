"""Prepare executions from schema-3 cases that the loader already resolved."""

import math
import re
import time
import hashlib
import json
import sys
import subprocess
import numpy as np

from contextlib import contextmanager
from collections import Counter
from collections.abc import Iterable, Callable
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from uuid import uuid4
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime, timezone
from typing import Literal

from src.ai.comprehension.orchestration import run_dcl_repetition_for_result
from src.ai.comprehension.storage import save_dcl_repetition_outcome
from src.ai.diagnostics import build_session_diagnostic_contexts
from src.ai.reporting.orchestration import run_reporting_for_dcl_repetition
from src.ai.reporting.storage import save_data_explanation_reporting_outcome
from src.common.time import timestamp
from src.experiments.assertions import validate_assertions, evaluate_declared_assertions
from src.experiments.models import Experiment, ReusedWeighting, ResolvedExecutionInputs
from src.experiments.progress import (
    AIRepetitionProgress,
    ProgressEvent,
    RunProgress,
    StageName,
    StageProgress,
)
from src.experiments.results import RunResult
from src.experiments.storage import output_flags, save_run, write_json_atomic, ArtifactWriteError
from src.experiments.comparison import evaluate_comparisons
from src.mcdm.enums import (
    WeightingMethod, RankingMethod, MeanMethod, GroupPowerMethod,
    EmptyGroupPolicy, OnExceedancePolicy, PairwisePreference,
)
from src.mcdm.service import MCDMService, validate_weight_vector, build_ranking_weighting
from src.mcdm.models.computation import EntityCalculation, CriterionWeight, StageError, EntityMetadata, WeightingSource
from src.mcdm.models.domain import Alternative, CriterionValue
from src.mcdm.reference import DERIVATIONS
from src.ai.component import AIComponent


ROOT = Path(__file__).resolve().parents[2]

VALID_WEIGHTING_METHODS = [WeightingMethod.AHP]
VALID_RANKING_METHODS = [RankingMethod.TOPSIS]

ALLOWED_POLICIES = {
    "alternative_alignment": {"require_identical_ids", "common_ids"},
    "criterion_alignment": {"require_identical_ids", "common_ids"},
}
ALLOWED_SCOPES = {"participant", "group", "session"}

IMPLEMENTED_AI_LAYERS = frozenset({
    "data_comprehension",
    "data_explanation_reporting",
})

ProgressCallback = Callable[[ProgressEvent], None]

def _hash(payload):
    return hashlib.sha256(payload).hexdigest()

def _hash_json(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    return _hash(encoded)

@dataclass(frozen=True)
class PreparedRun:
    """One execution, with independent inputs and links to other executions.

    key is deterministic within a plan, not a persistent UUID. The executor must
    assign a fresh run_id when saving an actual attempt. reference_runs are used
    after calculation; dependencies must finish before this calculation starts.
    """

    key: str
    case_key: str
    repetition: int
    seed: int
    experiment: Experiment
    requested: bool
    dependencies: tuple[str, ...]
    reference_runs: dict[str, str]

def _case_key(experiment):
    key = experiment.variant_id or experiment.session.id
    if not isinstance(key, str) or not key or key != experiment.session.id:
        raise ValueError("variant_id and session.id must identify the same resolved case.")
    if key != f"{experiment.id}/{experiment.case_metadata.get('case_id')}":
        raise ValueError(f"Invalid resolved case key: {key!r}.")
    return key


def _nonnegative_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def _unique_ids(items, label):
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {label} IDs.")
    return set(ids)

def _require_fields(value, *, required, optional=(), label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")

    missing = set(required) - set(value)
    unknown = set(value) - set(required) - set(optional)

    if missing or unknown:
        raise ValueError(
            f"{label}: missing fields {sorted(missing)}; "
            f"unsupported fields {sorted(unknown)}."
        )


def _require_ids(value, *, label, allowed=None):
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str) or not item
            for item in value
        )
    ):
        raise ValueError(
            f"{label} must be a nonempty list of strings."
        )

    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicate IDs.")

    if allowed is not None and not set(value) <= set(allowed):
        raise ValueError(f"{label} contains unknown IDs.")

    return value


def _require_dependency(reference, declared):
    if (
        not isinstance(reference, str)
        or not reference
        or reference not in declared
    ):
        raise ValueError(
            f"Reference {reference!r} must appear in "
            "result_dependencies."
        )

def _validate_runtime_declarations(experiment):
    case = experiment.case_metadata

    if not isinstance(case, dict):
        raise ValueError("Case metadata must be an object.")

    declared = case.get("result_dependencies", [])

    if not isinstance(declared, list):
        raise ValueError("result_dependencies must be a list.")

    if declared:
        _require_ids(
            declared,
            label="result_dependencies",
        )

    declared = set(declared)

    operations = case.get("runtime_operations", [])

    if not isinstance(operations, list):
        raise ValueError("runtime_operations must be a list.")

    full_ids = {
        criterion.id
        for criterion in experiment.session.criteria
    }
    ranking_ids = set(full_ids)

    alternative_ids = {
        alternative.id
        for alternative in experiment.session.alternatives
    }

    overrides = {}

    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("Every runtime operation must be an object.")

        kind = operation.get("type")

        if kind == "remove_criteria":
            _require_fields(
                operation,
                required={
                    "type",
                    "criterion_ids",
                    "target",
                    "preserve_preference_evidence",
                },
                label=kind,
            )

            if (
                operation["target"] != "ranking"
                or operation["preserve_preference_evidence"] is not True
            ):
                raise ValueError("Criterion removal must target ranking and preserve preference evidence.")

            removed = _require_ids(
                operation["criterion_ids"],
                label="criterion_ids",
                allowed=ranking_ids,
            )

            ranking_ids -= set(removed)

            if not ranking_ids:
                raise ValueError("At least one ranking criterion must remain.")

        elif kind == "set_criterion_weights":
            _require_fields(
                operation,
                required={"type", "scopes", "weights"},
                label=kind,
            )

            scopes = _require_ids(
                operation["scopes"],
                label="Weight override scopes",
                allowed=ALLOWED_SCOPES,
            )

            weights = operation["weights"]

            if (
                not isinstance(weights, dict)
                or set(weights) != ranking_ids
            ):
                raise ValueError("Explicit weights must cover the current ranking criteria.")

            ordered_ids = sorted(ranking_ids)

            validate_weight_vector(
                [weights[cid] for cid in ordered_ids],
                expected_count=len(ordered_ids),
                label="Declared explicit weights",
            )

            for scope in scopes:
                overrides[scope] = set(weights)

        elif kind == "remove_winner":
            _require_fields(
                operation,
                required={
                    "type",
                    "reference",
                    "tie_policy",
                    "on_empty_alternatives",
                },
                optional={"tie_atol"},
                label=kind,
            )

            _require_dependency(
                operation["reference"],
                declared,
            )

            if operation["tie_policy"] != "remove_all_winners":
                raise ValueError("Unsupported tie policy.")

            if operation["on_empty_alternatives"] != "error":
                raise ValueError("Unsupported empty-alternative policy.")

            if not _nonnegative_number(
                operation.get("tie_atol", 1e-12)
            ):
                raise ValueError("tie_atol must be finite and nonnegative.")

            # The surviving IDs require the reference ranking.
            alternative_ids = None

        elif kind == "inherit_alternatives":
            _require_fields(
                operation,
                required={"type", "reference"},
                label=kind,
            )

            _require_dependency(
                operation["reference"],
                declared,
            )

            # The dependency's resolved rows are not available yet.
            alternative_ids = None

        elif kind == "remove_alternatives":
            _require_fields(
                operation,
                required={"type", "alternative_ids"},
                label=kind,
            )

            removed = _require_ids(
                operation["alternative_ids"],
                label="alternative_ids",
                allowed=alternative_ids,
            )

            if alternative_ids is not None:
                alternative_ids -= set(removed)

                if not alternative_ids:
                    raise ValueError("At least one alternative must remain.")

        elif kind == "add_alternatives":
            _require_fields(
                operation,
                required={"type", "alternatives"},
                label=kind,
            )

            additions = operation["alternatives"]

            if not isinstance(additions, list) or not additions:
                raise ValueError("alternatives must be a nonempty list.")

            added_ids = set()

            for item in additions:
                _require_fields(
                    item,
                    required={"id", "name", "values"},
                    optional={"description"},
                    label="Added alternative",
                )

                for field in ("id", "name"):
                    if (
                        not isinstance(item[field], str)
                        or not item[field]
                    ):
                        raise ValueError(f"Alternative {field} must be a nonempty string.")

                if (
                    "description" in item
                    and not isinstance(item["description"], str)
                ):
                    raise ValueError("Alternative description must be a string.")

                aid = item["id"]

                if (
                    aid in added_ids
                    or (
                        alternative_ids is not None
                        and aid in alternative_ids
                    )
                ):
                    raise ValueError(f"Duplicate alternative ID: {aid!r}.")

                values = item["values"]

                if (
                    not isinstance(values, dict)
                    or set(values) != full_ids
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        for value in values.values()
                    )
                ):
                    raise ValueError("Added alternatives require finite values for every preference criterion.")

                added_ids.add(aid)

            if alternative_ids is not None:
                alternative_ids |= added_ids

        else:
            raise ValueError(f"Unsupported runtime operation: {kind!r}.")

    # Reject overrides invalidated by a later criterion removal.
    for scope, ids in overrides.items():
        if ids != ranking_ids:
            raise ValueError(f"{scope} override does not match the final ranking criteria.")

    reuse = case.get("reuse", {})

    _require_fields(
        reuse,
        required=set(),
        optional={"criterion_weights"},
        label="reuse",
    )

    if "criterion_weights" not in reuse:
        return

    rule = reuse["criterion_weights"]

    _require_fields(
        rule,
        required={"reference", "criterion_alignment"},
        optional={"scopes"},
        label="criterion_weights reuse",
    )

    _require_dependency(rule["reference"], declared)

    scopes = _require_ids(
        rule.get("scopes", ["session"]),
        label="Reuse scopes",
        allowed=ALLOWED_SCOPES,
    )

    alignment = rule["criterion_alignment"]

    if alignment not in (
        "require_identical_ids",
        "restrict_and_renormalize",
    ):
        raise ValueError("Unsupported criterion alignment policy.")

    if (
        alignment == "require_identical_ids"
        and ranking_ids != full_ids
    ):
        raise ValueError("Removed ranking criteria require restrict_and_renormalize reuse.")

    if set(scopes) & set(overrides):
        raise ValueError("A scope cannot both reuse and explicitly override criterion weights.")
    
def _validate_case(experiment):
    """Check executable settings and numerical input shape without calculating."""
    if experiment.schema_version != "3.0" or experiment.judgment_encoding != "pairwise_preference":
        raise ValueError("prepare_runs requires resolved schema-3 preference-code cases.")
    if experiment.execution.variations or experiment.execution.variants:
        raise ValueError("Variations must already be expanded by the loader.")
    defaults = experiment.defaults
    if defaults is None:
        raise ValueError("A loaded case must include its defaults.")
    if defaults.pipeline not in {"mcdm", "ai-mcdm"}:
        raise ValueError("Unsupported experiment pipeline.")

    expected_ai_enabled = defaults.pipeline == "ai-mcdm"

    if defaults.ai.enabled is not expected_ai_enabled:
        raise ValueError("Pipeline and defaults.ai.enabled disagree.")
    if experiment.ai != defaults.ai:
        raise ValueError("Experiment AI settings and defaults.ai disagree.")
    
    if isinstance(defaults.repetition, bool) or not isinstance(defaults.repetition, int) or defaults.repetition < 1:
        raise ValueError("repetitions must be a positive integer.")
    if isinstance(defaults.seed, bool) or not isinstance(defaults.seed, int) or defaults.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if defaults.seed != experiment.execution.random_seed:
        raise ValueError("defaults.seed and execution.random_seed disagree.")
    process = experiment.processing
    if process.weighting_method.method not in VALID_WEIGHTING_METHODS:
        raise ValueError("Unsupported weighting method.")
    if process.ranking_method.method not in VALID_RANKING_METHODS:
        raise ValueError("Unsupported ranking method.")
    weighting = process.weighting_method.parameters
    ranking = process.ranking_method.parameters
    if (not isinstance(weighting, dict) or set(weighting) - {"derivation"}
            or weighting.get("derivation", "geometric") not in DERIVATIONS):
        raise ValueError("Unsupported AHP parameters or derivation.")
    if (not isinstance(ranking, dict) or set(ranking) - {"normalization"}
            or ranking.get("normalization", "vector") != "vector"):
        raise ValueError("Only vector TOPSIS normalization is supported.")
    aggregation = process.aggregation_method
    if (aggregation.within_group != MeanMethod.GEOMETRIC_MEAN
            or aggregation.between_groups != MeanMethod.WEIGHTED_GEOMETRIC_MEAN
            or aggregation.group_power_normalization != GroupPowerMethod.REPRESENTED_GROUPS
            or aggregation.empty_group_policy != EmptyGroupPolicy.EXCLUDED):
        raise ValueError("Unsupported aggregation configuration for the reference engine.")
    consistency = process.consistency_method
    if (not _nonnegative_number(consistency.threshold)
            or consistency.on_exceedance != OnExceedancePolicy.RETAIN_AND_FLAG):
        raise ValueError("Unsupported consistency configuration.")
    if consistency.threshold != experiment.execution.consistency_threshold:
        raise ValueError("Processing and execution consistency thresholds disagree.")

    session = experiment.session
    cids = _unique_ids(session.criteria, "criterion")
    gids = _unique_ids(session.stakeholder_groups, "group")
    qids = _unique_ids(session.questions, "question")
    _unique_ids(session.participants, "participant")
    _unique_ids(session.alternatives, "alternative")
    if not 2 <= len(cids) <= 15 or not session.alternatives or not session.participants:
        raise ValueError("AHP needs 2–15 criteria, participants, and at least one alternative.")
    if any(c.direction not in {"max", "min"} for c in session.criteria):
        raise ValueError("Criterion directions must be max or min.")
    pairs = [frozenset((q.left_criterion_id, q.right_criterion_id)) for q in session.questions]
    if len(pairs) != len(set(pairs)) or set(pairs) != {frozenset(p) for p in combinations(cids, 2)}:
        raise ValueError("Questions must cover every criterion pair exactly once.")
    represented = set()
    for participant in session.participants:
        if participant.stakeholder_group_id not in gids:
            raise ValueError("A participant references an unknown group.")
        represented.add(participant.stakeholder_group_id)
        response_ids = [r.question_id for r in participant.responses]
        if len(response_ids) != len(qids) or set(response_ids) != qids:
            raise ValueError("Each participant must answer every question exactly once.")
        if any(not isinstance(r.preference, PairwisePreference) for r in participant.responses):
            raise ValueError("Participant judgments must remain PairwisePreference values.")
    powers = {g.id: g.voting_power for g in session.stakeholder_groups}
    if any(not _nonnegative_number(v) for v in powers.values()) or not any(powers[g] > 0 for g in represented):
        raise ValueError("Represented groups require positive total voting power.")
    for alternative in session.alternatives:
        values = alternative.criterion_values
        if len(values) != len(cids) or {v.criterion_id for v in values} != cids:
            raise ValueError("Alternative values must cover every criterion exactly once.")
        if any(isinstance(v.value, bool) or not isinstance(v.value, (int, float))
               or not math.isfinite(v.value) for v in values):
            raise ValueError("Alternative values must be finite numbers.")

    _validate_runtime_declarations(experiment)

    return validate_assertions(experiment.expected_assertions, criterion_ids=cids, group_ids=gids)


def _references(experiment, assertion_references):
    """Collect reporting references without turning them into execution edges."""
    references = []
    if experiment.baseline_case is not None:
        references.append(experiment.baseline_case)
    if not isinstance(experiment.comparisons, list):
        raise ValueError("Comparisons must be a list.")
    comparison_ids = set()
    for comparison in experiment.comparisons:
        if not isinstance(comparison, dict):
            raise ValueError("Each comparison must be an object.")
        identifier = comparison.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in comparison_ids:
            raise ValueError("Comparison IDs must be unique nonempty strings within a case.")
        comparison_ids.add(identifier)
        metrics = comparison.get("metrics")
        if (not isinstance(metrics, list) or not metrics
                or any(not isinstance(m, str) for m in metrics)
                or set(metrics) - {"weight_delta", "score_delta", "winner_changed", "spearman_rho", "kendall_tau_b"}):
            raise ValueError("A comparison specifies unsupported metrics.")
        if not _nonnegative_number(comparison.get("tie_atol", 1e-12)):
            raise ValueError("Comparison tie_atol must be finite and nonnegative.")
        for field, allowed in ALLOWED_POLICIES.items():
            policy = comparison.get(field, "require_identical_ids")
            if not isinstance(policy, str) or policy not in allowed:
                raise ValueError(
                    f"Unsupported comparison {field} policy: {policy!r}."
                )
        if not isinstance(comparison.get("rerank_common_alternatives", False), bool):
            raise ValueError("Comparison rerank_common_alternatives must be a boolean.")
        references.append(comparison.get("reference"))
    references.extend(assertion_references)
    if any(not isinstance(ref, str) or not ref for ref in references):
        raise ValueError("References must be nonempty case keys.")
    return tuple(dict.fromkeys(references))


def _normalize_selector(value):
    family, separator, case = value.partition("/")
    match = re.fullmatch(r"S0*(\d+)", family, re.IGNORECASE)
    if match:
        family = f"S{int(match[1]):02}"
    return family + separator + case

def _optional_provenance(
    label,
    action,
    notes,
    *,
    errors,
):
    try:
        return action()

    except errors as error:
        notes.append(
            f"{label} unavailable: "
            f"{type(error).__name__}: {error}"
        )
        return None

def _run_key(case_key, repetition):
    return f"{case_key}#r{repetition:03d}"

def _dependency_is_ready(result: RunResult) -> bool:
    if result.execution.get("numerical_status") != "completed":
        return False

    result.validate_numerical_complete()
    return True

def _validate_execution_order(runs):
    positions = {}

    for index, run in enumerate(runs):
        if run.key in positions:
            raise ValueError(
                f"Duplicate prepared execution key: {run.key!r}"
            )
        positions[run.key] = index

    for run in runs:
        for dependency_key in run.dependencies:
            if dependency_key not in positions:
                raise ValueError(
                    f"{run.key}: missing dependency "
                    f"{dependency_key!r}."
                )

            if positions[dependency_key] >= positions[run.key]:
                raise ValueError(
                    f"{run.key}: dependency {dependency_key!r} "
                    "must appear before its consumer."
                )
            
    
def _create_run_result(run: PreparedRun) -> RunResult:
    experiment = run.experiment

    return RunResult(
        experiment_id=experiment.id,
        case_key=run.case_key,
        name=experiment.name,
        run_id=uuid4().hex,
        variant_id=experiment.variant_id,
        status="failed",
        inputs=experiment.to_dict(),
        execution={
            "execution_key": run.key,
            "repetition": run.repetition,
            "seed": run.seed,
            "dependencies": list(run.dependencies),
            "reference_runs": dict(run.reference_runs),
            "created_at": timestamp(),
            "attempt_started_at": None,
            "attempt_completed_at": None,
            "attempt_duration_seconds": None,
            "numerical_started_at": None,
            "numerical_completed_at": None,
            "numerical_duration_seconds": None,
            "numerical_status": "not_run",
            "messages": [],
            "execution_scope": "numerical",
            "ai_status": "not_requested",
        },
    )

def provenance(experiment, inputs):
    notes = []

    def git_output(*arguments):
        try:
            process = subprocess.run(
                [
                    "git",
                    "--no-optional-locks",
                    "-C",
                    str(ROOT),
                    *arguments,
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            notes.append(f"Git metadata unavailable: {type(error).__name__}")
            return None

        if process.returncode != 0:
            notes.append(f"Git command failed: {' '.join(arguments)}")
            return None

        return process.stdout.strip()

    revision = git_output("rev-parse", "HEAD")
    worktree_status = git_output(
        "status",
        "--porcelain",
        "--untracked-files=normal",
    )

    source = Path(experiment.source_path).resolve()

    resources = {
        "experiment": source,
        "criteria": (
            source.parent / experiment.resources.criteria
        ).resolve(),
        "alternatives": (
            source.parent / experiment.resources.alternatives
        ).resolve(),
        "dataset": (
            source.parent / experiment.resources.dataset
        ).resolve(),
    }

    resource_hashes = {}

    for name, path in resources.items():
        resource_hashes[name] = {
            "path": str(path),
            "sha256": _optional_provenance(
                f"Resource hash: {name}",
                lambda path=path: _hash(path.read_bytes()),
                notes,
                errors=(OSError,),
            ),
        }

    discovered = _optional_provenance(
        "Source-code discovery",
        lambda: list((ROOT / "src").rglob("*.py")),
        notes,
        errors=(OSError,),
    )

    code_paths = set(discovered or [])

    code_paths.update(
        ROOT / name
        for name in (
            "main.py",
            "requirements.txt",
            "requirements-dev.txt",
        )
    )

    code_hashes = {}

    for path in sorted(code_paths):
        code_hashes[path.relative_to(ROOT).as_posix()] = (
            _optional_provenance(
                f"Code hash: {path.name}",
                lambda path=path: _hash(path.read_bytes()),
                notes,
                errors=(OSError,),
            )
        )

    dependency_versions = {}

    for package in (
        "numpy",
        "scipy",
        "pydantic",
        "pyDecision",
        "openai",
    ):
        dependency_versions[package] = _optional_provenance(
            f"Package version: {package}",
            lambda package=package: version(package),
            notes,
            errors=(PackageNotFoundError,),
        )

    requested_input_hash = _optional_provenance(
        "Requested input hash",
        lambda: _hash_json(inputs),
        notes,
        errors=(TypeError, ValueError),
    )
    
    return {
        "requested_input_hash": requested_input_hash,
        "resource_hashes": resource_hashes,
        "code_hashes": code_hashes,
        "code_revision": revision,
        "dirty_worktree": (
            None
            if worktree_status is None
            else bool(worktree_status)
        ),
        "python": sys.version,
        "dependencies": dependency_versions,
        "seed": experiment.execution.random_seed,
        "processing": inputs["processing"],
        "engine": {
            "weighting": "pyDecision",
            "ranking": "pyDecision",
            "aggregation": "local judgment-matrix aggregation",
            "two_criterion_ahp": "local analytical convention",
        },
        "notes": notes,
    }

@contextmanager
def entity_stage(calculation, name):
    if name not in {"weighting", "ranking"}:
        raise ValueError(f"Unknown entity stage: {name!r}")

    record = getattr(calculation.metadata.stages, name)

    if record.status != "not_run":
        raise ValueError(
            f"Stage {name!r} has already started."
        )

    record.status = "running"
    record.started_at = timestamp()
    started = time.perf_counter()

    try:
        yield

    except Exception as error:
        record.status = "failed"
        record.error = StageError(
            type=type(error).__name__,
            message=str(error),
        )
        raise

    else:
        record.status = "completed"

    finally:
        record.completed_at = timestamp()
        record.duration_seconds = (
            time.perf_counter() - started
        )

def calculate_entity(
    *,
    entity_id,
    scope,
    pairwise_matrix,
    decision_matrix,
    session,
    processing,
    metadata: EntityMetadata | None = None,
    reused_weighting: ReusedWeighting | None = None,
    ranking_criterion_ids=None,
    criterion_weight_override=None,
    on_created: Callable[[EntityCalculation], None] | None = None,
) -> EntityCalculation:
    calculation = EntityCalculation(
        entity_id=entity_id,
        scope=scope,
        pairwise_matrix=pairwise_matrix.tolist(),
        metadata=(
            EntityMetadata()
            if metadata is None
            else deepcopy(metadata)
        ),
    )

    if on_created is not None:
        on_created(calculation)

    with entity_stage(calculation, "weighting"):
        if reused_weighting is None:
            weighting = MCDMService.generate_weights(
                session=session,
                pairwise_matrix=pairwise_matrix,
                config=processing.weighting_method,
                consistency=processing.consistency_method,
            )

            calculation.metadata.weighting_source = WeightingSource(kind="calculated")

        else:
            if (
                pairwise_matrix.shape
                != reused_weighting.pairwise_matrix.shape
                or not np.allclose(
                    pairwise_matrix,
                    reused_weighting.pairwise_matrix,
                    atol=1e-12,
                    rtol=0,
                )
            ):
                raise ValueError(
                    "Current preferences do not match the reused weights."
                )

            weighting = reused_weighting.weighting.model_copy(
                deep=True
            )

            calculation.metadata.weighting_source = WeightingSource(
                kind="reused",
                source_run_id=reused_weighting.source_run_id,
                source_case_key=reused_weighting.source_case_key,
            )

        # Preserve successful weighting before projection or ranking begins.
        calculation.weighting = weighting

    with entity_stage(calculation, "ranking"):
        full_ids = [
            criterion.id
            for criterion in session.criteria
        ]

        selected_ids = (
            full_ids
            if ranking_criterion_ids is None
            else list(ranking_criterion_ids)
        )

        (
            ranking_weighting,
            applied_weights,
            weight_source,
        ) = build_ranking_weighting(
            weighting,
            full_criterion_ids=full_ids,
            ranking_criterion_ids=selected_ids,
            override=criterion_weight_override,
        )

        positions = {
            criterion_id: index
            for index, criterion_id in enumerate(full_ids)
        }

        selected_positions = [
            positions[criterion_id]
            for criterion_id in selected_ids
        ]

        # Preserve the full preference session. This view supplies only
        # the selected criterion columns to the ranking service.
        ranking_view = session.model_copy(update={
            "criteria": [
                session.criteria[index]
                for index in selected_positions
            ],
        })

        calculation.metadata.ranking_criterion_ids = selected_ids
        calculation.metadata.applied_criterion_weights = applied_weights
        calculation.metadata.ranking_weight_source = weight_source

        calculation.ranking = MCDMService.generate_ranking(
            session=ranking_view,
            decision_matrix=decision_matrix[
                :, selected_positions
            ],
            weighting=ranking_weighting,
            config=processing.ranking_method,
        )

    return calculation

def prepare_runs(
    experiments: Iterable[Experiment],
    scenario: str | Iterable[str] | None = None,
) -> list[PreparedRun]:
    """Select and validate loaded cases, then expand independent executions.

    A selector may be a family (S5/S05), exact case key, or a list of selectors.
    Baselines, comparison/assertion references, and explicit result_dependencies
    are included transitively. input_from is already resolved and is ignored.

    Repetition i uses base_seed+i-1. References pair with repetition i, or use the
    sole reference run when its case has one repetition. Other counts are rejected.
    Comparisons are post-calculation tasks and may have reciprocal references.
    Only result_dependencies are topologically ordered and checked for cycles.
    """
    # 1. Index the resolved cases and validate their execution configuration.
    catalog, references, dependencies = {}, {}, {}
    for experiment in experiments:
        key = _case_key(experiment)
        if key in catalog:
            raise ValueError(f"Duplicate case key: {key!r}.")
        try:
            assertion_references = _validate_case(experiment)
            references[key] = _references(experiment, assertion_references)
            hard = experiment.case_metadata.get("result_dependencies", [])
            if not isinstance(hard, list) or any(not isinstance(v, str) or not v for v in hard):
                raise ValueError("result_dependencies must be a list of case keys.")
            dependencies[key] = tuple(dict.fromkeys(hard))
        except ValueError as error:
            raise ValueError(f"{key}: {error}") from error
        catalog[key] = experiment
    if not catalog:
        raise ValueError("No loaded experiment cases were supplied.")

    # 2. Validate every reference and order only actual result dependencies.
    for key in catalog:
        for reference in references[key] + dependencies[key]:
            if reference not in catalog:
                raise ValueError(f"{key}: missing referenced case {reference!r}.")
        for comparison in catalog[key].comparisons:
            other = catalog[comparison["reference"]]

            for entities, policy_field in (
                ("criteria", "criterion_alignment"),
                ("alternatives", "alternative_alignment"),
            ):
                own_ids = {
                    entity.id
                    for entity in getattr(catalog[key].session, entities)
                }
                other_ids = {
                    entity.id
                    for entity in getattr(other.session, entities)
                }
                policy = comparison.get(
                    policy_field, "require_identical_ids"
                )

                # Strict comparisons require the same entities on both sides.
                # common_ids comparisons may use different entity sets;
                # their intersection is determined during comparison execution.
                if policy == "require_identical_ids" and own_ids != other_ids:
                    raise ValueError(
                        f"{key}: comparison {comparison['id']!r} "
                        f"requires identical {entities} IDs."
                    )
    order, visiting, visited = [], set(), set()

    def order_case(key):
        if key in visiting:
            raise ValueError(f"Result dependency cycle at {key!r}.")
        if key in visited:
            return
        visiting.add(key)
        for dependency in sorted(dependencies[key]):
            order_case(dependency)
        visiting.remove(key)
        visited.add(key)
        order.append(key)

    for key in sorted(catalog):
        order_case(key)

    # 3. Match each selector; a typo must not silently select nothing.
    if scenario is None:
        requested = set(catalog)
    else:
        if isinstance(scenario, str):
            selectors = [scenario]
        else:
            try:
                selectors = list(scenario)
            except TypeError as error:
                raise ValueError("scenario must be a selector or an iterable of selectors.") from error
        if not selectors or any(not isinstance(s, str) or not s for s in selectors):
            raise ValueError("At least one nonempty scenario selector is required.")
        requested = set()
        for selector in selectors:
            normalized = _normalize_selector(selector)
            matches = {key for key, e in catalog.items()
                       if _normalize_selector(key) == normalized
                       or _normalize_selector(e.id) == normalized}
            if not matches:
                raise ValueError(f"No experiment or case matches {selector!r}.")
            requested.update(matches)

    # 4. Include all references transitively. The visited set makes comparison
    #    cycles harmless; these edges do not dictate calculation order.
    included, pending = set(), list(requested)
    while pending:
        key = pending.pop()
        if key not in included:
            included.add(key)
            pending.extend(references[key] + dependencies[key])

    # 5. Resolve repeated references explicitly rather than choosing arbitrary
    #    trials or recalculating a baseline separately for every consumer.
    def reference_run(source, target, repetition):
        source_count = catalog[source].defaults.repetition
        target_count = catalog[target].defaults.repetition
        if target_count == 1:
            return _run_key(target, 1)
        if target_count == source_count:
            return _run_key(target, repetition)
        raise ValueError(
            f"{source}: cannot pair {source_count} repetitions with {target!r} "
            f"({target_count} repetitions); use equal counts or one reference run."
        )

    # 6. Build independent runs. Do not recreate cases or regenerate participants.
    #    The executor assigns persistent IDs and evaluates the declared checks.
    runs = []
    for key in order:
        if key not in included:
            continue
        experiment = catalog[key]
        for repetition in range(1, experiment.defaults.repetition + 1):
            seed = experiment.defaults.seed + repetition - 1
            inputs = deepcopy(experiment)
            inputs.execution.random_seed = seed
            inputs.defaults.seed = seed
            runs.append(PreparedRun(
                key=_run_key(key, repetition), case_key=key,
                repetition=repetition, seed=seed, experiment=inputs,
                requested=key in requested,
                dependencies=tuple(reference_run(key, dep, repetition) for dep in dependencies[key]),
                reference_runs={ref: reference_run(key, ref, repetition) for ref in references[key]},
            ))
    return runs

def _build_reused_weighting(
    calculation: EntityCalculation,
    *,
    source: RunResult,
    source_criterion_ids: list[str],
    target_criterion_ids: list[str],
) -> ReusedWeighting:
    # Align the stored source matrix with the target criterion order.
    positions = {
        criterion_id: index
        for index, criterion_id in enumerate(source_criterion_ids)
    }

    order = [
        positions[criterion_id]
        for criterion_id in target_criterion_ids
    ]

    matrix = np.asarray(
        calculation.pairwise_matrix,
        dtype=float,
    )

    expected_shape = (
        len(source_criterion_ids),
        len(source_criterion_ids),
    )

    if (
        matrix.shape != expected_shape
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError(
            f"Invalid source matrix for "
            f"{calculation.scope}/{calculation.entity_id}."
        )

    matrix = matrix[np.ix_(order, order)].copy()

    weighting = calculation.weighting.model_copy(deep=True)

    weights_by_id = {
        item.criterion_id: item
        for item in weighting.criterion_weights
    }

    if (
        len(weights_by_id) != len(weighting.criterion_weights)
        or set(weights_by_id) != set(target_criterion_ids)
    ):
        raise ValueError(
            f"Source weights do not cover the target criteria for "
            f"{calculation.scope}/{calculation.entity_id}."
        )

    values = np.asarray([
        weights_by_id[criterion_id].weight
        for criterion_id in target_criterion_ids
    ])

    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0)
        or not np.isclose(
            values.sum(),
            1.0,
            atol=1e-10,
            rtol=0,
        )
    ):
        raise ValueError(
            f"Invalid source weights for "
            f"{calculation.scope}/{calculation.entity_id}."
        )

    # Keep stored weights in target criterion order as well.
    weighting.criterion_weights = [
        weights_by_id[criterion_id].model_copy(deep=True)
        for criterion_id in target_criterion_ids
    ]

    return ReusedWeighting(
        weighting=weighting,
        pairwise_matrix=matrix,
        source_run_id=source.run_id,
        source_case_key=source.case_key,
    )

def resolve_execution_inputs(
    run: PreparedRun,
    *,
    dependency_results: dict[str, RunResult] | None = None,
) -> ResolvedExecutionInputs:
    """Resolve one independent attempt's runtime interventions and weight reuse.

    Operations run in JSON order on a deep copy. Result-consuming operations
    require declared, usable dependency results. Ranking-only interventions do
    not rewrite AHP questions, judgments, matrices, or consistency evidence;
    the executor applies their separate ranking dimensions and scoped weights.
    Metadata records each intervention and any reused result's provenance.
    """
    dependencies = dependency_results or {}
    experiment = deepcopy(run.experiment)

    resolved = ResolvedExecutionInputs(
        experiment=experiment,
        metadata={"runtime_operations": []},
        ranking_criterion_ids=[c.id for c in experiment.session.criteria],
    )

    def get_dependency(reference):
        if (not isinstance(reference, str)
                or reference not in experiment.case_metadata.get("result_dependencies", [])
                or reference not in dependencies):
            raise ValueError(
                f"{run.key}: reference {reference!r} must be "
                "a declared, supplied result dependency."
            )

        source = dependencies[reference]

        if source.case_key != reference:
            raise ValueError("Dependency reference identity mismatch.")

        if not _dependency_is_ready(source):
            raise ValueError(
                f"Dependency {reference!r} has no usable calculations."
            )

        return source

    def validate_ids(values, known, label):
        """Reject ambiguous, repeated, or unknown removal/selection IDs."""
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) for value in values)
                or len(values) != len(set(values)) or not set(values) <= set(known)):
            raise ValueError(f"{label} must contain unique known IDs.")

    def validate_alternatives(alternatives):
        """Check the complete preference dimension set before changing inputs."""
        ids = [a.id for a in alternatives]
        criteria = {c.id for c in experiment.session.criteria}
        if not alternatives or len(ids) != len(set(ids)):
            raise ValueError("Runtime alternatives must be nonempty with unique IDs.")
        for alternative in alternatives:
            values = alternative.criterion_values
            if (len(values) != len(criteria) or {v.criterion_id for v in values} != criteria
                    or any(not math.isfinite(v.value) for v in values)):
                raise ValueError("Runtime alternatives need finite values for every criterion.")

    # Resolve runtime operations in their declared order.
    operations = experiment.case_metadata.get(
        "runtime_operations", []
    )

    if not isinstance(operations, list):
        raise ValueError("runtime_operations must be a list.")

    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("Each runtime operation must be an object.")

        kind = operation.get("type")
        if kind == "remove_criteria":
            # S07 omits ranking columns only. Original preferences and their
            # consistency diagnostics remain available in the saved inputs.
            if set(operation) - {"type", "criterion_ids", "target", "preserve_preference_evidence"}:
                raise ValueError("Unsupported remove_criteria fields.")
            if (operation.get("target") != "ranking"
                    or operation.get("preserve_preference_evidence") is not True):
                raise ValueError("Criterion removal must preserve preference evidence and target ranking.")
            removed = operation.get("criterion_ids")
            validate_ids(removed, resolved.ranking_criterion_ids, "criterion_ids")
            retained = [cid for cid in resolved.ranking_criterion_ids if cid not in removed]
            if not retained:
                raise ValueError("Criterion removal must leave at least one ranking criterion.")
            resolved.ranking_criterion_ids = retained
            resolved.metadata["runtime_operations"].append({
                "type": kind, "removed_criterion_ids": list(removed),
                "remaining_criterion_ids": list(retained), "preserve_preference_evidence": True,
            })
            continue

        if kind == "set_criterion_weights":
            # S08 supplies applied TOPSIS weights at selected scopes without
            # replacing any calculated AHP weights or consistency evidence.
            if set(operation) - {"type", "scopes", "weights"}:
                raise ValueError("Unsupported set_criterion_weights fields.")
            selected_scopes = operation.get("scopes")
            validate_ids(selected_scopes, ALLOWED_SCOPES, "Weight override scopes")
            weights = operation.get("weights")
            if (not isinstance(weights, dict) or set(weights) != set(resolved.ranking_criterion_ids)
                    or any(not _nonnegative_number(value) for value in weights.values())
                    or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-10, rel_tol=0)):
                raise ValueError("Explicit weights must cover ranking criteria and sum to one.")
            for scope in selected_scopes:
                resolved.criterion_weight_overrides[scope] = deepcopy(weights)
            resolved.metadata["runtime_operations"].append({
                "type": kind, "scopes": list(selected_scopes), "weights": deepcopy(weights),
            })
            continue

        if kind == "add_alternatives":
            # S09 adds complete performance rows; it does not change criteria,
            # stakeholder judgments, or any existing alternative's values.
            if set(operation) - {"type", "alternatives"}:
                raise ValueError("Unsupported add_alternatives fields.")
            additions = operation.get("alternatives")
            if not isinstance(additions, list) or not additions:
                raise ValueError("add_alternatives requires a nonempty alternatives list.")
            parsed = []
            criteria = {c.id for c in experiment.session.criteria}
            for item in additions:
                if not isinstance(item, dict) or set(item) - {"id", "name", "description", "values"}:
                    raise ValueError("Invalid added alternative fields.")
                values = item.get("values")
                if (not isinstance(values, dict) or set(values) != criteria
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in values.values())):
                    raise ValueError("Added alternatives need finite values for every criterion.")
                if any(not isinstance(item.get(k), str) or not item[k] for k in ("id", "name")):
                    raise ValueError("Added alternatives require nonempty IDs and names.")
                parsed.append(Alternative(
                    id=item["id"], name=item["name"], description=item.get("description", ""),
                    criterion_values=[CriterionValue(criterion_id=cid, value=value)
                                      for cid, value in values.items()],
                ))
            combined = experiment.session.alternatives + parsed
            validate_alternatives(combined)
            experiment.session.alternatives = combined
            resolved.metadata["runtime_operations"].append({
                "type": kind, "added_alternative_ids": [a.id for a in parsed],
            })
            continue

        if kind == "inherit_alternatives":
            # Copy the dependency's resolved rows (including earlier additions),
            # rather than reloading its original resource file.
            if set(operation) - {"type", "reference"}:
                raise ValueError("Unsupported inherit_alternatives fields.")
            source = get_dependency(operation.get("reference"))
            source_criteria = {c["id"]: c for c in source.inputs["session"]["criteria"]}
            current_criteria = {c.id: c.model_dump(mode="json") for c in experiment.session.criteria}
            if source_criteria != current_criteria:
                raise ValueError("Inherited alternatives require identical criterion definitions.")
            inherited = [Alternative.model_validate(deepcopy(a))
                         for a in source.inputs["session"]["alternatives"]]
            validate_alternatives(inherited)
            experiment.session.alternatives = inherited
            resolved.metadata["runtime_operations"].append({
                "type": kind, "reference": source.case_key, "source_run_id": source.run_id,
                "inherited_alternative_ids": [a.id for a in inherited],
            })
            continue

        if kind == "remove_alternatives":
            # Remove named rows from the current, already transformed set.
            # This makes S09's add -> inherit -> remove sequence reversible.
            if set(operation) - {"type", "alternative_ids"}:
                raise ValueError("Unsupported remove_alternatives fields.")
            removed = operation.get("alternative_ids")
            validate_ids(removed, [a.id for a in experiment.session.alternatives], "alternative_ids")
            remaining = [a for a in experiment.session.alternatives if a.id not in removed]
            validate_alternatives(remaining)
            experiment.session.alternatives = remaining
            resolved.metadata["runtime_operations"].append({
                "type": kind, "removed_alternative_ids": list(removed),
                "remaining_alternative_ids": [a.id for a in remaining],
            })
            continue

        if kind != "remove_winner":
            raise NotImplementedError(
                f"Runtime operation {operation.get('type')!r} "
                "is not implemented."
            )

        # S06 removes the entire reference winner set using its stored scores,
        # so ties are not broken arbitrarily by presentation order.
        if set(operation) - {"type", "reference", "tie_policy", "tie_atol", "on_empty_alternatives"}:
            raise ValueError("Unsupported remove_winner fields.")
        if operation.get("tie_policy") != "remove_all_winners":
            raise ValueError("Unsupported winner-removal tie policy.")

        if operation.get("on_empty_alternatives") != "error":
            raise ValueError("Unsupported empty-alternative policy.")

        tolerance = operation.get("tie_atol", 1e-12)

        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError("tie_atol must be finite and nonnegative.")

        source = get_dependency(operation.get("reference"))
        rankings = source.mcdm.session.ranking.rankings

        if not rankings:
            raise ValueError("Dependency has no session ranking.")

        scores = [
            item.closeness_coefficient
            for item in rankings
        ]

        if not all(math.isfinite(score) for score in scores):
            raise ValueError("Dependency ranking contains invalid scores.")

        best_score = max(scores)

        winner_ids = {
            item.alternative_id
            for item in rankings
            if abs(item.closeness_coefficient - best_score)
            <= tolerance
        }

        current_ids = {
            alternative.id
            for alternative in experiment.session.alternatives
        }

        # This initial handler implements the S06 design:
        # start from the same alternative set as the reference.
        reference_ids = {
            alternative["id"]
            for alternative in source.inputs["session"]["alternatives"]
        }

        if current_ids != reference_ids:
            raise ValueError(
                "remove_winner currently requires the same starting "
                "alternative IDs as its reference."
            )

        remaining = [
            alternative
            for alternative in experiment.session.alternatives
            if alternative.id not in winner_ids
        ]

        if not remaining:
            raise ValueError(
                "Removing the winner set would leave no alternatives."
            )

        experiment.session.alternatives = remaining

        resolved.metadata["runtime_operations"].append({
            "type": "remove_winner",
            "reference": source.case_key,
            "source_run_id": source.run_id,
            "removed_alternative_ids": sorted(winner_ids),
            "remaining_alternative_ids": [
                alternative.id
                for alternative in remaining
            ],
            "tie_atol": tolerance,
        })

    # Reject weight maps made stale by a later dimension change. Operation order
    # is explicit: remove criteria before setting weights for the reduced set.
    for weights in resolved.criterion_weight_overrides.values():
        if set(weights) != set(resolved.ranking_criterion_ids):
            raise ValueError("Weight overrides must match the final ranking criteria.")
    resolved.metadata["ranking_criterion_ids"] = list(resolved.ranking_criterion_ids)

    # Resolve requested artifact reuse after the alternative/criterion operations.
    # Reuse retains full original AHP evidence; the executor restricts and
    # renormalizes applied ranking weights over ranking_criterion_ids.
    reuse = experiment.case_metadata.get("reuse", {})

    if not isinstance(reuse, dict):
        raise ValueError("reuse must be an object.")

    if set(reuse) - {"criterion_weights"}:
        raise NotImplementedError("Unsupported reuse artifact.")

    if "criterion_weights" not in reuse:
        return resolved

    rule = reuse["criterion_weights"]

    if not isinstance(rule, dict):
        raise ValueError("criterion_weights reuse must be an object.")

    if set(rule) - {"reference", "criterion_alignment", "scopes"}:
        raise ValueError("Unsupported criterion-weight reuse fields.")

    scopes = rule.get("scopes", ["session"])

    if (
        not isinstance(scopes, list)
        or not scopes
        or any(
            not isinstance(scope, str)
            or scope not in ALLOWED_SCOPES
            for scope in scopes
        )
    ):
        raise ValueError("Reuse scopes must be a nonempty list containing participant, group, or session.")

    if len(scopes) != len(set(scopes)):
        raise ValueError("Reuse scopes must not contain duplicates.")

    alignment = rule.get("criterion_alignment")
    if not isinstance(alignment, str) or alignment not in {"require_identical_ids", "restrict_and_renormalize"}:
        raise ValueError("Unsupported criterion alignment policy.")
    if set(scopes) & set(resolved.criterion_weight_overrides):
        raise ValueError("A scope cannot both reuse and explicitly override criterion weights.")
    if (alignment == "require_identical_ids"
            and set(resolved.ranking_criterion_ids) != {c.id for c in experiment.session.criteria}):
        raise ValueError("Removed ranking criteria require restrict_and_renormalize reuse.")

    source = get_dependency(rule["reference"])

    current_ids = [
        criterion.id
        for criterion in experiment.session.criteria
    ]

    source_ids = [
        criterion["id"]
        for criterion in source.inputs["session"]["criteria"]
    ]

    if (
        len(source_ids) != len(set(source_ids))
        or set(current_ids) != set(source_ids)
    ):
        raise ValueError(
            "Reused weights require identical criterion IDs."
        )

    # Require unchanged weighting and consistency configurations.
    current_processing = experiment.to_dict()["processing"]
    source_processing = source.inputs["processing"]

    for name in ("weighting_method", "consistency_method"):
        if current_processing[name] != source_processing[name]:
            raise ValueError(
                f"Cannot reuse weights after changing {name}."
            )

    # Match criterion definitions by ID rather than list position.
    current_criteria = {
        item["id"]: item
        for item in experiment.to_dict()["session"]["criteria"]
    }

    source_criteria = {
        item["id"]: item
        for item in source.inputs["session"]["criteria"]
    }

    if current_criteria != source_criteria:
        raise ValueError(
            "Cannot reuse weights after changing criterion definitions."
        )

    source_collections = {
        "participant": source.mcdm.participants,
        "group": source.mcdm.groups,
    }

    represented_groups = {
        participant.stakeholder_group_id
        for participant in experiment.session.participants
    }

    target_ids_by_scope = {
        "participant": [
            participant.id
            for participant in experiment.session.participants
        ],
        "group": [
            group.id
            for group in experiment.session.stakeholder_groups
            if group.id in represented_groups
        ],
    }

    resolved.metadata["weight_reuse"] = []

    for scope in scopes:
        if scope == "session":
            calculation = source.mcdm.session

            if calculation is None:
                raise ValueError(
                    "Dependency is missing its session calculation."
                )

            if calculation.scope != "session":
                raise ValueError(
                    "Dependency has an invalid session calculation."
                )

            pairs = [
                (
                    experiment.session.id,
                    calculation,
                )
            ]

        else:
            calculations = source_collections[scope]

            source_by_id = {
                calculation.entity_id: calculation
                for calculation in calculations
            }

            if len(source_by_id) != len(calculations):
                raise ValueError(
                    f"Duplicate source {scope} calculation IDs."
                )

            if any(
                calculation.scope != scope
                for calculation in calculations
            ):
                raise ValueError(
                    f"Incorrect scope in source {scope} calculations."
                )

            target_ids = target_ids_by_scope[scope]

            if len(target_ids) != len(set(target_ids)):
                raise ValueError(
                    f"Duplicate target {scope} IDs."
                )

            # This initial policy requires exact entity matching.
            if set(source_by_id) != set(target_ids):
                raise ValueError(
                    f"Full {scope} reuse requires identical "
                    "source and target entity IDs."
                )

            pairs = [
                (entity_id, source_by_id[entity_id])
                for entity_id in target_ids
            ]

        for target_id, calculation in pairs:
            key = (scope, target_id)

            resolved.reused_weightings[key] = (
                _build_reused_weighting(
                    calculation,
                    source=source,
                    source_criterion_ids=source_ids,
                    target_criterion_ids=current_ids,
                )
            )

            resolved.metadata["weight_reuse"].append({
                "scope": scope,
                "target_entity_id": target_id,
                "source_entity_id": calculation.entity_id,
                "reference": source.case_key,
                "source_run_id": source.run_id,
                "criterion_alignment": alignment,
                "ranking_criterion_ids": list(resolved.ranking_criterion_ids),
            })

    return resolved

def execute_run(
    run: PreparedRun,
    *,
    dependency_results: dict[str, RunResult] | None = None
) -> RunResult:
    result = _create_run_result(run)

    result.execution["attempt_started_at"] = timestamp()
    attempt_started = time.perf_counter()
    numerical_started = None

    try:
        result.execution["resolution_status"] = "not_run"
        result.execution["provenance_status"] = "running"

        # Capture provenance for the original prepared inputs.
        result.provenance = provenance(
            run.experiment,
            result.inputs,
        )

        result.provenance["dependency_run_ids"] = {
            reference: dependency.run_id
            for reference, dependency in (
                dependency_results or {}
            ).items()
        }

        notes = result.provenance["notes"]

        result.execution["provenance_status"] = (
            "incomplete" if notes else "completed"
        )

        # Resolve runtime operations before inspecting their outputs.
        result.execution["resolution_status"] = "running"

        resolved = resolve_execution_inputs(
            run,
            dependency_results=dependency_results,
        )

        experiment = resolved.experiment
        session = experiment.session
        processing = experiment.processing

        result.inputs = experiment.to_dict()
        result.execution["input_resolution"] = resolved.metadata
        result.execution["resolution_status"] = "completed"

        # Hash the actual resolved inputs and resolution metadata.
        result.provenance["resolved_input_hash"] = (
            _optional_provenance(
                "Resolved input hash",
                lambda: _hash_json(result.inputs),
                notes,
                errors=(TypeError, ValueError),
            )
        )

        result.provenance["resolution_hash"] = (
            _optional_provenance(
                "Resolution hash",
                lambda: _hash_json(resolved.metadata),
                notes,
                errors=(TypeError, ValueError),
            )
        )

        result.execution["provenance_status"] = (
            "incomplete" if notes else "completed"
        )

        # Numerical timing begins only when calculation begins.
        result.execution["numerical_status"] = "running"
        result.execution["numerical_started_at"] = timestamp()
        numerical_started = time.perf_counter()

        decision_matrix = MCDMService.create_decision_matrix(
            session
        )

        # Participants
        for participant in session.participants:
            matrix = MCDMService.create_participant_matrix(
                session,
                participant
            )

            calculation = calculate_entity(
                entity_id=participant.id,
                scope="participant",
                pairwise_matrix=matrix,
                decision_matrix=decision_matrix,
                session=session,
                processing=processing,
                metadata=EntityMetadata(
                    group_id=participant.stakeholder_group_id,
                ),
                reused_weighting=resolved.reused_weightings.get(
                    ("participant", participant.id)
                ),
                ranking_criterion_ids=resolved.ranking_criterion_ids,
                criterion_weight_override=resolved.criterion_weight_overrides.get("participant"),
                on_created=result.mcdm.participants.append,
            )

        # Stakeholder groups
        for group in session.stakeholder_groups:
            members = [
                participant
                for participant in session.participants
                if participant.stakeholder_group_id == group.id
            ]

            if not members:
                # This branch assumes your configured policy is "exclude".
                continue

            session_aggregation = (
                MCDMService.create_session_aggregation(
                    session=session,
                    config=processing.aggregation_method,
                )
            )

            matrix = MCDMService.create_stakeholder_group_matrix(
                session,
                members,
                config=processing.aggregation_method,
            )

            calculation = calculate_entity(
                entity_id=group.id,
                scope="group",
                pairwise_matrix=matrix,
                decision_matrix=decision_matrix,
                session=session,
                processing=processing,
                metadata=EntityMetadata(
                    participant_count=len(members),
                    voting_power=group.voting_power,
                    effective_voting_power=(
                        session_aggregation.effective_voting_powers[group.id]
                    )
                ),
                reused_weighting=resolved.reused_weightings.get(
                    ("group", group.id)
                ),
                ranking_criterion_ids=resolved.ranking_criterion_ids,
                criterion_weight_override=resolved.criterion_weight_overrides.get("group"),
                on_created=result.mcdm.groups.append
            )

        # Complete session.
        matrix = MCDMService.create_session_matrix(
            session,
            config=processing.aggregation_method,
        )

        result.mcdm.session = calculate_entity(
            entity_id=session.id,
            scope="session",
            pairwise_matrix=matrix,
            decision_matrix=decision_matrix,
            session=session,
            processing=processing,
            reused_weighting=resolved.reused_weightings.get(
                ("session", session.id)
            ),
            ranking_criterion_ids=resolved.ranking_criterion_ids,
            criterion_weight_override=resolved.criterion_weight_overrides.get("session"),
            on_created=lambda calculation: setattr(result.mcdm, "session", calculation)
        )

        result.set_status("calculated")
        result.execution["numerical_status"] = "completed"

    except Exception as error:
        result.set_status("failed")

        result.execution["numerical_status"] = (
            "not_run"
            if numerical_started is None
            else "failed"
        )

        if result.execution.get("resolution_status") == "running":
            result.execution["resolution_status"] = "failed"

        result.execution["messages"].append(
            f"{type(error).__name__}: {error}"
        )
    finally:
        result.execution["attempt_completed_at"] = timestamp()
        result.execution["attempt_duration_seconds"] = (
            time.perf_counter() - attempt_started
        )

        if numerical_started is not None:
            result.execution["numerical_completed_at"] = timestamp()
            result.execution["numerical_duration_seconds"] = (
                time.perf_counter() - numerical_started
            )

    return result

def _execute_numerical_runs(
    runs: Iterable[PreparedRun],
    *,
    on_result: Callable[[PreparedRun, RunResult], None] | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[RunResult]:
    runs = list(runs)
    _validate_execution_order(runs)

    completed: dict[str, RunResult] = {}
    results: list[RunResult] = []

    for index, run in enumerate(runs, start=1):
        run_started = time.perf_counter()

        if on_progress is not None:
            on_progress(RunProgress(
                stage="numerical",
                execution_key=run.key,
                state="started",
                current=index,
                total=len(runs),
            ))

        blocked_by = {
            key: completed[key]
            for key in run.dependencies
            if not _dependency_is_ready(completed[key])
        }

        if blocked_by:
            result = _create_run_result(run)
            result.set_status("blocked")

            result.execution["numerical_status"] = "not_run"
            result.execution["dependency_status"] = "blocked"

            result.execution["blocked_by"] = {
                key: {
                    "run_id": dependency.run_id,
                    "status": dependency.status,
                    "numerical_status": dependency.execution.get(
                        "numerical_status"
                    ),
                }
                for key, dependency in blocked_by.items()
            }

            result.execution["messages"].append(
                "Numerical execution was skipped because required "
                "dependency results were unavailable."
            )

        else:
            dependency_results = {
                completed[key].case_key: completed[key]
                for key in run.dependencies
            }

            if len(dependency_results) != len(run.dependencies):
                raise ValueError(
                    "Multiple dependencies resolve to the same case key."
                )

            result = execute_run(
                run,
                dependency_results=dependency_results,
            )

            result.execution["dependency_status"] = "satisfied"

        result.execution["dependency_run_ids"] = {
            key: completed[key].run_id
            for key in run.dependencies
        }

        result.execution.setdefault(
            "assertions_status", "not_run"
        )
        result.execution.setdefault(
            "comparisons_status", "not_run"
        )
        result.execution.setdefault(
            "validation_status", "not_run"
        )

        completed[run.key] = result
        results.append(result)

        # Artifact failures must propagate to the caller.
        if on_result is not None:
            on_result(run, result)

        if on_progress is not None:
            numerical_status = result.execution.get(
                "numerical_status"
            )
            state = (
                "blocked"
                if result.status == "blocked"
                else "completed"
                if numerical_status == "completed"
                else "failed"
            )
            on_progress(RunProgress(
                stage="numerical",
                execution_key=run.key,
                state=state,
                current=index,
                total=len(runs),
                duration_seconds=(
                    time.perf_counter() - run_started
                ),
                detail=f"numerical status: {numerical_status}",
            ))

    return results

def finalize_run_validation(
    run: PreparedRun,
    result: RunResult,
) -> None:
    if result.execution.get("numerical_status") != "completed":
        raise ValueError(
            "Cannot finalize validation without completed "
            "numerical results."
        )

    result.validate_numerical_complete()

    def index_records(records, label):
        if not isinstance(records, list):
            raise ValueError(f"{label} must be a list.")

        indexed = {}

        for record in records:
            if not isinstance(record, dict):
                raise ValueError(
                    f"Each {label} record must be an object."
                )

            identifier = record.get("id")

            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in indexed
            ):
                raise ValueError(
                    f"{label} IDs must be unique nonempty strings."
                )

            indexed[identifier] = record

        return indexed

    declared_assertions = {
        spec["id"]: spec
        for spec in run.experiment.expected_assertions
    }

    assertion_records = index_records(
        result.assertions,
        "Assertion",
    )

    if set(assertion_records) != set(declared_assertions):
        raise ValueError(
            "Assertion results do not cover exactly the "
            "declared assertions."
        )

    declared_comparisons = {
        spec["id"]: spec
        for spec in run.experiment.comparisons
    }

    if not isinstance(result.comparison, dict):
        raise ValueError("Missing comparison-stage results.")

    comparison_records = index_records(
        result.comparison.get("items"),
        "Comparison",
    )

    if set(comparison_records) != set(declared_comparisons):
        raise ValueError(
            "Comparison results do not cover exactly the "
            "declared comparisons."
        )

    allowed_assertion_statuses = {
        "passed",
        "failed",
        "error",
        "deferred",
    }

    for record in assertion_records.values():
        if record.get("status") not in allowed_assertion_statuses:
            raise ValueError(
                "An assertion has an invalid result status."
            )

    required_assertion_failures = [
        identifier
        for identifier, record in assertion_records.items()
        if (
            declared_assertions[identifier].get(
                "severity", "error"
            ) == "error"
            and record["status"] != "passed"
        )
    ]

    comparison_failures = [
        identifier
        for identifier, record in comparison_records.items()
        if record.get("status") != "completed"
    ]

    warnings = [
        identifier
        for identifier, record in assertion_records.items()
        if (
            declared_assertions[identifier].get(
                "severity", "error"
            ) == "warning"
            and record["status"] != "passed"
        )
    ]

    result.execution["validation_summary"] = {
        "assertion_status_counts": dict(Counter(
            record["status"]
            for record in assertion_records.values()
        )),
        "required_assertion_failures": required_assertion_failures,
        "comparison_failures": comparison_failures,
        "warnings": warnings,
    }

    passed = not (
        required_assertion_failures
        or comparison_failures
    )

    result.execution["validation_status"] = (
        "passed" if passed else "failed"
    )

    if not passed:
        identifiers = (
            required_assertion_failures
            + comparison_failures
        )

        result.execution["messages"].append(
            "Required validation did not pass: "
            + ", ".join(identifiers)
        )

    result.set_status("passed" if passed else "failed")


def execute_experiments(
    runs,
    output,
    *,
    ai_component: AIComponent | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[Path, list[RunResult]]:
    active_stage: StageName | None = None
    active_stage_started: float | None = None

    def emit(event: ProgressEvent) -> None:
        if on_progress is not None:
            on_progress(event)

    def start_stage(stage: StageName) -> None:
        nonlocal active_stage, active_stage_started
        active_stage = stage
        active_stage_started = time.perf_counter()
        emit(StageProgress(stage=stage, state="started"))

    def finish_stage(
        state: Literal[
            "completed",
            "partial",
            "failed",
            "skipped",
        ],
        *,
        detail: str | None = None,
    ) -> None:
        nonlocal active_stage, active_stage_started

        if active_stage is None:
            return

        stage = active_stage
        started = active_stage_started
        active_stage = None
        active_stage_started = None

        emit(StageProgress(
            stage=stage,
            state=state,
            duration_seconds=(
                time.perf_counter() - started
                if started is not None
                else None
            ),
            detail=detail,
        ))

    if output is None:
        raise ValueError("An output directory is required.")

    runs = list(runs)
    if not runs:
        raise ValueError("No prepared runs were supplied.")

    ai_runs = [
        run
        for run in runs
        if run.experiment.ai.enabled
    ]
    if ai_runs and ai_component is None:
        raise ValueError("AI-enabled experiments require an initialized AI component.")

    if ai_runs:
        assert ai_component is not None

        for run in ai_runs:
            request = run.experiment.ai
            requested_layers = set(request.layers)

            if (
                "data_explanation_reporting"
                in requested_layers
                and "data_comprehension"
                not in requested_layers
            ):
                raise ValueError(
                    f"{run.key}: "
                    "data_explanation_reporting requires "
                    "data_comprehension."
                )

            unsupported_layers = (
                requested_layers
                - IMPLEMENTED_AI_LAYERS
            )

            if unsupported_layers:
                names = ", ".join(
                    sorted(unsupported_layers)
                )

                raise ValueError(
                    f"{run.key}: requested AI layers "
                    f"are not implemented: {names}."
                )

            if (
                request.provider is not None
                and request.provider
                != ai_component.config.provider.value
            ):
                raise ValueError(
                    f"{run.key}: requested provider "
                    f"{request.provider!r} does not match "
                    "the configured provider "
                    f"{ai_component.config.provider.value!r}."
                )

            if (
                request.model is not None
                and request.model
                != ai_component.config.model
            ):
                raise ValueError(
                    f"{run.key}: requested model "
                    f"{request.model!r} does not match "
                    "the configured model "
                    f"{ai_component.config.model!r}."
                )

            if request.parameters:
                raise ValueError(
                    f"{run.key}: experiment-specific AI "
                    "parameters are not implemented."
                )

            if request.prompt_overrides:
                raise ValueError(
                    f"{run.key}: prompt overrides are "
                    "not implemented."
                )

            if request.reported_result_interventions:
                raise ValueError(
                    f"{run.key}: reported-result AI "
                    "interventions are not implemented "
                    "for this execution path."
                )

    _validate_execution_order(runs)

    batch_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid4().hex
    )

    batch_directory = (
        Path(output).expanduser().resolve() / batch_id
    )

    try:
        batch_directory.mkdir(
            parents=True,
            exist_ok=False,
        )
    except OSError as error:
        raise ArtifactWriteError(
            f"Could not create batch directory "
            f"{batch_directory}: {error}"
        ) from error

    if ai_component is not None:
        preflight = ai_component.preflight.model_copy(
            update={"batch_id": batch_id}
        )

        write_json_atomic(
            batch_directory / "ai/preflight.json",
            preflight,
        )

    manifest = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "status": "running",
        "phase": "numerical",
        "created_at": timestamp(),
        "completed_at": None,
        "planned_execution_keys": [
            run.key for run in runs
        ],
        "runs": {
            run.key: {
                "case_key": run.case_key,
                "run_id": None,
                "status": "pending",
                "directory": None,
            }
            for run in runs
        },
    }

    manifest_path = batch_directory / "manifest.json"

    def checkpoint(run, result):
        entry = save_run(
            batch_directory,
            run,
            result,
        )

        # Update the manifest only after the run files were saved.
        manifest["runs"][run.key] = entry
        manifest["updated_at"] = timestamp()

        write_json_atomic(manifest_path, manifest)

    def run_artifact_directory(run) -> Path:
        entry = manifest["runs"][run.key]
        relative = entry.get("directory")

        if not isinstance(relative, str) or not relative:
            raise ArtifactWriteError("Run artifacts must be checkpointed before AI execution.")

        directory = (
            batch_directory / relative
        ).resolve()

        if not directory.is_relative_to(batch_directory):
            raise ArtifactWriteError("Run artifact directory escaped the batch.")

        return directory

    try:
        write_json_atomic(manifest_path, manifest)

        start_stage("numerical")

        # Pass 1: calculate and checkpoint every planned execution.
        results = _execute_numerical_runs(
            runs,
            on_result=checkpoint,
            on_progress=on_progress,
        )

        completed = {
            result.execution["execution_key"]: result
            for result in results
        }

        numerical_counts = Counter(
            result.execution.get("numerical_status", "unknown")
            for result in results
        )
        numerical_completed = numerical_counts.get("completed", 0)
        numerical_stage_state = (
            "completed"
            if numerical_completed == len(results)
            else "partial"
            if numerical_completed
            else "failed"
        )
        run_label = "run" if len(results) == 1 else "runs"
        finish_stage(
            numerical_stage_state,
            detail=(
                f"{len(results)} {run_label} processed; "
                + ", ".join(
                    f"{status}={count}"
                    for status, count in sorted(
                        numerical_counts.items()
                    )
                )
            ),
        )

        start_stage("validation")

        manifest["phase"] = "validation"
        write_json_atomic(manifest_path, manifest)

        # Pass 2: all numerical references are now available.
        for index, run in enumerate(runs, start=1):
            result = completed[run.key]
            validation_run_started = time.perf_counter()

            if (
                result.execution.get("numerical_status")
                != "completed"
            ):
                emit(RunProgress(
                    stage="validation",
                    execution_key=run.key,
                    state="skipped",
                    current=index,
                    total=len(runs),
                    duration_seconds=0.0,
                    detail="numerical calculation did not complete",
                ))
                continue

            emit(RunProgress(
                stage="validation",
                execution_key=run.key,
                state="started",
                current=index,
                total=len(runs),
            ))

            result.execution["validation_status"] = "running"
            result.execution["validation_started_at"] = timestamp()
            validation_started = time.perf_counter()

            try:
                result.execution["assertions_status"] = "running"

                result.assertions = evaluate_declared_assertions(
                    run,
                    result,
                    completed,
                    stage="calculation",
                )

                result.execution["comparisons_status"] = "running"

                result.comparison = {
                    "items": evaluate_comparisons(
                        run,
                        result,
                        completed,
                    )
                }

                result.execution["comparisons_status"] = "completed"

                result.assertions.extend(
                    evaluate_declared_assertions(
                        run,
                        result,
                        completed,
                        stage="comparison",
                    )
                )

                result.execution["assertions_status"] = "completed"

                finalize_run_validation(run, result)

            except Exception as error:
                for field in (
                    "assertions_status",
                    "comparisons_status",
                    "validation_status",
                ):
                    if result.execution.get(field) == "running":
                        result.execution[field] = "failed"

                result.execution["validation_status"] = "failed"
                result.execution["messages"].append(
                    f"Validation error: "
                    f"{type(error).__name__}: {error}"
                )
                result.set_status("failed")

            finally:
                result.execution["validation_completed_at"] = (
                    timestamp()
                )
                result.execution["validation_duration_seconds"] = (
                    time.perf_counter() - validation_started
                )

            # Keep storage errors outside the validation handler.
            checkpoint(run, result)

            validation_status = result.execution.get(
                "validation_status"
            )
            emit(RunProgress(
                stage="validation",
                execution_key=run.key,
                state=(
                    "completed"
                    if validation_status == "passed"
                    else "failed"
                ),
                current=index,
                total=len(runs),
                duration_seconds=(
                    time.perf_counter()
                    - validation_run_started
                ),
                detail=f"validation status: {validation_status}",
            ))

        validation_counts = Counter(
            result.execution.get("validation_status", "not_run")
            for result in results
        )
        validation_passed = validation_counts.get("passed", 0)
        validation_stage_state = (
            "completed"
            if validation_passed == len(results)
            else "partial"
            if validation_passed
            else "failed"
        )
        finish_stage(
            validation_stage_state,
            detail="; ".join(
                f"{status}={count}"
                for status, count in sorted(
                    validation_counts.items()
                )
            ),
        )

        manifest["phase"] = "ai"
        write_json_atomic(manifest_path, manifest)

        if ai_runs:
            start_stage("ai")

        for index, run in enumerate(runs, start=1):
            result = completed[run.key]

            if not run.experiment.ai.enabled:
                result.execution["ai_status"] = "not_requested"
                if ai_runs:
                    emit(RunProgress(
                        stage="ai",
                        execution_key=run.key,
                        state="skipped",
                        current=index,
                        total=len(runs),
                        detail="AI not requested",
                    ))
                continue

            if (
                result.execution.get("numerical_status")
                != "completed"
            ):
                result.execution["ai_status"] = "blocked"
                result.execution["messages"].append(
                    "AI execution was blocked because numerical "
                    "calculation did not complete."
                )
                checkpoint(run, result)
                emit(RunProgress(
                    stage="ai",
                    execution_key=run.key,
                    state="blocked",
                    current=index,
                    total=len(runs),
                    detail="numerical calculation did not complete",
                ))
                continue

            if (
                result.execution.get("validation_status")
                != "passed"
            ):
                result.execution["ai_status"] = "blocked"
                result.execution["messages"].append(
                    "AI execution was blocked because deterministic "
                    "validation did not pass."
                )
                checkpoint(run, result)
                emit(RunProgress(
                    stage="ai",
                    execution_key=run.key,
                    state="blocked",
                    current=index,
                    total=len(runs),
                    detail="deterministic validation did not pass",
                ))
                continue

            if "data_comprehension" not in run.experiment.ai.layers:
                result.execution["ai_status"] = "not_requested"
                checkpoint(run, result)
                emit(RunProgress(
                    stage="ai",
                    execution_key=run.key,
                    state="skipped",
                    current=index,
                    total=len(runs),
                    detail="data comprehension layer not requested",
                ))
                continue

            ai_scope = [
                "data_comprehension.session_understanding",
                "data_comprehension.stakeholder_group_profiles",
                "data_comprehension.session_profile",
            ]

            reporting_requested = (
                "data_explanation_reporting"
                in run.experiment.ai.layers
            )

            if reporting_requested:
                ai_scope.append(
                    "data_explanation_reporting"
                )

            result.execution["ai_scope"] = ai_scope

            result.execution["ai_status"] = "running"
            result.execution["ai_started_at"] = timestamp()
            ai_started = time.perf_counter()
            result.execution["ai_repetitions"] = {}

            emit(RunProgress(
                stage="ai",
                execution_key=run.key,
                state="started",
                current=index,
                total=len(runs),
                detail=(
                    f"{run.experiment.ai.repetitions} "
                    "repetition(s)"
                ),
            ))

            flags = output_flags(
                run.experiment.output
            )

            run_directory = run_artifact_directory(run)

            repetition_statuses = []
            session_diagnostics = (
                build_session_diagnostic_contexts(result)
            )

            for ai_repetition in range(
                1,
                run.experiment.ai.repetitions + 1,
            ):
                repetition_started = time.perf_counter()
                emit(AIRepetitionProgress(
                    execution_key=run.key,
                    repetition=ai_repetition,
                    total=run.experiment.ai.repetitions,
                    state="started",
                ))

                dcl_outcome = run_dcl_repetition_for_result(
                    backend=ai_component.backend,
                    result=result,
                    ai_repetition=ai_repetition,
                    max_retries=(
                        ai_component.config.max_retries
                    ),
                    retry_base_seconds=(
                        ai_component.config.retry_base_seconds
                    ),
                    retry_max_wait_seconds=(
                        ai_component.config.retry_max_wait_seconds
                    ),
                    on_progress=emit,
                    session_diagnostics=(
                        session_diagnostics
                    ),
                    session_understanding_limitations=(
                        "Authored scenario information and "
                        "configured model inputs are not "
                        "independently verified empirical evidence.",
                    ),
                    group_profile_limitations=(
                        "Modeled group results do not establish "
                        "unanimity, motivation, or broader "
                        "population preferences.",
                    ),
                    session_profile_limitations=(
                        "Aggregate modeled results do not by "
                        "themselves establish consensus, actual "
                        "influence, causal effects, robustness, "
                        "feasibility, or policy desirability.",
                    ),
                )

                dcl_artifact_root = save_dcl_repetition_outcome(
                    run_directory=run_directory,
                    outcome=dcl_outcome,
                    save_ai_calls=flags["save_ai_calls"],
                    save_ai_responses=flags[
                        "save_ai_responses"
                    ],
                )

                reporting_outcome = None
                reporting_artifact_root = None
                reporting_status = "not_requested"

                if (
                    "data_explanation_reporting"
                    in run.experiment.ai.layers
                ):
                    reporting_outcome = (
                        run_reporting_for_dcl_repetition(
                            backend=ai_component.backend,
                            result=result,
                            dcl_outcome=dcl_outcome,
                            max_retries=(
                                ai_component.config.max_retries
                            ),
                            retry_base_seconds=(
                                ai_component.config
                                .retry_base_seconds
                            ),
                            retry_max_wait_seconds=(
                                ai_component.config
                                .retry_max_wait_seconds
                            ),
                            on_progress=emit,
                        )
                    )

                    reporting_status = reporting_outcome.status

                    if reporting_outcome.stage_outcome is not None:
                        reporting_artifact_root = (
                            save_data_explanation_reporting_outcome(
                                run_directory=run_directory,
                                outcome=(
                                    reporting_outcome.stage_outcome
                                ),
                                artifact=reporting_outcome.artifact,
                                save_ai_calls=(
                                    flags["save_ai_calls"]
                                ),
                                save_ai_responses=(
                                    flags["save_ai_responses"]
                                ),
                            )
                        )

                if reporting_status == "not_requested":
                    repetition_status = dcl_outcome.status

                elif (
                    dcl_outcome.status == "completed"
                    and reporting_status == "completed"
                ):
                    repetition_status = "completed"

                elif dcl_outcome.status == "failed":
                    repetition_status = "failed"

                else:
                    repetition_status = "partial"

                repetition_statuses.append(
                    repetition_status
                )

                result.execution["ai_repetitions"][
                    str(ai_repetition)
                ] = {
                    "status": repetition_status,
                    "stages": {
                        "session_understanding": (
                            dcl_outcome
                            .session_understanding_status
                        ),
                        "stakeholder_group_profiles": (
                            dcl_outcome.group_profiles_status
                        ),
                        "session_profile": (
                            dcl_outcome.session_profile_status
                        ),
                        "data_explanation_reporting": reporting_status,
                    },
                    "expected_group_ids": list(
                        dcl_outcome.expected_group_ids
                    ),
                    "successful_group_ids": [
                        artifact.request.group_id
                        for artifact
                        in dcl_outcome.state.group_profiles
                    ],
                    "failed_group_ids": list(
                        dcl_outcome.failed_group_ids
                    ),
                    "artifacts": {
                        "session_understanding": (
                            dcl_outcome.state.session_understanding
                            is not None
                        ),
                        "group_profile_count": len(
                            dcl_outcome.state.group_profiles
                        ),
                        "session_profile": (
                            dcl_outcome.state.session_profile
                            is not None
                        ),
                        "data_explanation_report": (
                            reporting_outcome is not None
                            and reporting_outcome.artifact is not None
                        ),
                    },
                    "call_count": len(dcl_outcome.calls),
                    "reporting_call_count": (
                        len(reporting_outcome.stage_outcome.calls)
                        if (
                            reporting_outcome is not None
                            and reporting_outcome.stage_outcome
                            is not None
                        )
                        else 0
                    ),
                    "artifact_directory": (
                        dcl_artifact_root
                        .relative_to(run_directory)
                        .as_posix()
                    ),
                    "reporting_artifact_directory": (
                        reporting_artifact_root
                        .relative_to(run_directory)
                        .as_posix()
                        if reporting_artifact_root is not None
                        else None
                    ),
                    "reporting_blocked_reason": (
                        reporting_outcome.blocked_reason
                        if reporting_outcome is not None
                        else None
                    ),
                }

                # Preserve completed repetition progress.
                checkpoint(run, result)

                emit(AIRepetitionProgress(
                    execution_key=run.key,
                    repetition=ai_repetition,
                    total=run.experiment.ai.repetitions,
                    state=repetition_status,
                    duration_seconds=(
                        time.perf_counter() - repetition_started
                    ),
                    detail=(
                        "session understanding="
                        f"{dcl_outcome.session_understanding_status}; "
                        "group profiles="
                        f"{dcl_outcome.group_profiles_status}; "
                        "session profile="
                        f"{dcl_outcome.session_profile_status}; "
                        f"{len(dcl_outcome.state.group_profiles)} "
                        "group profile(s) completed; "
                        f"{len(dcl_outcome.failed_group_ids)} failed"
                        "; data explanation reporting="
                        f"{reporting_status}"
                    ),
                ))

            if all(
                status == "completed"
                for status in repetition_statuses
            ):
                ai_status = "completed"
            elif any(
                status in {"completed", "partial"}
                for status in repetition_statuses
            ):
                ai_status = "partial"
            else:
                ai_status = "failed"

            result.execution["ai_status"] = ai_status
            result.execution["ai_completed_at"] = timestamp()
            result.execution["ai_duration_seconds"] = (
                time.perf_counter() - ai_started
            )

            checkpoint(run, result)

            emit(RunProgress(
                stage="ai",
                execution_key=run.key,
                state=result.execution["ai_status"],
                current=index,
                total=len(runs),
                duration_seconds=result.execution[
                    "ai_duration_seconds"
                ],
                detail=(
                    f"AI status: {result.execution['ai_status']}"
                ),
            ))

        if ai_runs:
            ai_counts = Counter(
                result.execution.get("ai_status", "not_requested")
                for run, result in zip(runs, results, strict=True)
                if run.experiment.ai.enabled
            )
            ai_completed = ai_counts.get("completed", 0)
            ai_partial = ai_counts.get("partial", 0)
            ai_stage_state = (
                "completed"
                if ai_completed == len(ai_runs)
                else "partial"
                if ai_completed or ai_partial
                else "failed"
            )
            finish_stage(
                ai_stage_state,
                detail="; ".join(
                    f"{status}={count}"
                    for status, count in sorted(ai_counts.items())
                ),
            )
        else:
            emit(StageProgress(
                stage="ai",
                state="skipped",
                detail="no AI-enabled runs selected",
            ))

        manifest["ai_status_counts"] = dict(
            Counter(
                result.execution.get(
                    "ai_status",
                    "not_requested",
                )
                for result in results
            )
        )
        
        manifest["phase"] = "finished"

        numerical_succeeded = all(
            result.status == "passed"
            for result in results
        )

        requested_ai_results = [
            result
            for run, result in zip(
                runs,
                results,
                strict=True,
            )
            if run.experiment.ai.enabled
        ]

        ai_succeeded = all(
            result.execution.get("ai_status")
            == "completed"
            for result in requested_ai_results
        )

        if not numerical_succeeded:
            manifest["status"] = "failed"
        elif requested_ai_results and not ai_succeeded:
            manifest["status"] = "partial"
        else:
            manifest["status"] = "completed"

        manifest["result_status_counts"] = dict(Counter(
            result.status
            for result in results
        ))
        manifest["completed_at"] = timestamp()

        write_json_atomic(manifest_path, manifest)

        return batch_directory, results

    except Exception as error:
        finish_stage(
            "failed",
            detail=f"{type(error).__name__}: {error}",
        )

        manifest["status"] = "aborted"
        manifest["completed_at"] = timestamp()
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }

        # Preserve the original exception if writing the failure
        # manifest also fails.
        try:
            write_json_atomic(manifest_path, manifest)
        except ArtifactWriteError:
            pass

        raise

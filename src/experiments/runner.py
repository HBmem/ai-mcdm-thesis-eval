"""Prepare executions from schema-3 cases that the loader already resolved."""

import math
import re
import time
import hashlib
import json
import sys
import subprocess
import numpy as np

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from uuid import uuid4
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

from src.common.time import timestamp
from src.experiments.assertions import validate_assertions
from src.experiments.models import Experiment, ReusedWeighting, ResolvedExecutionInputs
from src.experiments.results import RunResult
from src.mcdm.enums import (
    WeightingMethod, RankingMethod, MeanMethod, GroupPowerMethod,
    EmptyGroupPolicy, OnExceedancePolicy, PairwisePreference,
)
from src.mcdm.service import MCDMService
from src.mcdm.models.computation import EntityCalculation
from src.mcdm.models.domain import Alternative, CriterionValue
from src.mcdm.reference import DERIVATIONS


ROOT = Path(__file__).resolve().parent.parent

VALID_WEIGHTING_METHODS = [WeightingMethod.AHP]
VALID_RANKING_METHODS = [RankingMethod.TOPSIS]

ALLOWED_POLICIES = {
    "alternative_alignment": {"require_identical_ids", "common_ids"},
    "criterion_alignment": {"require_identical_ids", "common_ids"},
}
ALLOWED_SCOPES = {"participant", "group", "session"}

def _hash(payload):
    return hashlib.sha256(payload).hexdigest()

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


def _validate_case(experiment):
    """Check executable settings and numerical input shape without calculating."""
    if experiment.schema_version != "3.0" or experiment.judgment_encoding != "pairwise_preference":
        raise ValueError("prepare_runs requires resolved schema-3 preference-code cases.")
    if experiment.execution.variations or experiment.execution.variants:
        raise ValueError("Variations must already be expanded by the loader.")
    defaults = experiment.defaults
    if defaults is None:
        raise ValueError("A loaded case must include its defaults.")
    if defaults.pipeline != "mcdm" or defaults.ai.enabled is not False:
        raise ValueError("Only the numerical mcdm pipeline is currently supported.")
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


def _run_key(case_key, repetition):
    return f"{case_key}#r{repetition:03d}"

def _normalize_inputs(experiment):
    session = experiment.session
    return {
        "experiment_schema_version": experiment.schema_version,
        "research_purpose": experiment.research_purpose,
        "scenario": {
            "id": experiment.scenario_id,
            "description": experiment.scenario_description,
            "policy_question": experiment.scenario_policy_question
        },
        "criteria": [c.model_dump(mode="json") for c in session.criteria],
        "preference_criteria": [c.model_dump(mode="json") for c in session.criteria],
        "questions": [q.model_dump(mode="json") for q in session.questions],
        "groups": [g.model_dump(mode="json") for g in session.stakeholder_groups],
        "participants": [{"id": p.id, "group_id": p.stakeholder_group_id,
                              "judgments": {r.question_id: float(r.preference.ahp_scale if hasattr(r.preference, "ahp_scale") else r.preference) for r in p.responses}}
                             for p in session.participants],
        "alternatives": [{"id": a.id, "name": a.name, "description": a.description,
                              "values": {v.criterion_id: v.value for v in a.criterion_values}} for a in session.alternatives],
            "processing": {"weighting_method": experiment.processing.weighting_method.method.value,
                           "ranking_method": experiment.processing.ranking_method.method.value,
                           "wd": experiment.processing.weighting_method.parameters.get("wd", "geometric")},
            "consistency_threshold": experiment.execution.consistency_threshold,
            "random_seed": experiment.execution.random_seed, "weights": None
    }

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
            "numerical_status": "pending",
            "messages": [],
        },
    )

def provenance(experiment, inputs):
    def _hash(payload):
        return hashlib.sha256(payload).hexdigest()

    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

    hashes = {}
    source = Path(experiment.source_path)
    for label, path in (
        ("experiment", source),
        ("criteria", source.parent / experiment.resources.criteria),
        ("alternatives", source.parent / experiment.resources.alternatives),
        ("dataset", source.parent / experiment.resources.dataset)
    ):
        hashes[label] = _hash(path.read_bytes())

    versions = {}

    for package in ("numpy", "scipy", "pydantic", "pandas", "streamlit", "openai"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    # Include untracked source files: git revision alone cannot identify this workspace.
    code = {str(p.relative_to(ROOT)): _hash(p.read_bytes()) for p in sorted((ROOT / "src").rglob("*.py"))}

    for name in ("main.py", "app.py"):
        code[name] = _hash((ROOT / name).read_bytes())
    return {
        "resource_hashes": hashes,
        "input_hash": _hash(json.dumps(inputs, sort_keys=True, allow_nan=False).encode()),
        "code_revision": git("rev-parse", "HEAD"),
        "dirty_worktree": bool(git("status", "--porcelain")),
        "code_hashes": code,
        "python": sys.version,
        "dependencies": versions,
        "seed": experiment.execution.random_seed,
        "engine": "independent_numpy_reference",
        "poliinsight_validated": False
    }

def calculate_entity(
    *,
    entity_id,
    scope,
    pairwise_matrix,
    decision_matrix,
    session,
    processing,
    metadata=None,
    reused_weighting: ReusedWeighting | None = None,
    ranking_criterion_ids=None,
    criterion_weight_override=None,
) -> EntityCalculation:
    metadata = {} if metadata is None else dict(metadata)

    if reused_weighting is None:
        weighting = MCDMService.generate_weights(
            session=session,
            pairwise_matrix=pairwise_matrix,
            config=processing.weighting_method,
            consistency=processing.consistency_method,
        )

        metadata["weighting_source"] = "calculated"

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
            raise ValueError("Current preferences do not match the reused weights.")

        weighting = reused_weighting.weighting.model_copy(deep=True)

        metadata["weighting_source"] = {
            "type": "reused",
            "source_run_id": reused_weighting.source_run_id,
            "source_case_key": reused_weighting.source_case_key,
        }

    # Keep the full AHP weighting and consistency diagnostics as evidence.
    # Only the TOPSIS inputs are restricted or overridden by ranking operations.
    ranking_criterion_ids = (
        [c.id for c in session.criteria]
        if ranking_criterion_ids is None else ranking_criterion_ids
    )
    positions = {c.id: i for i, c in enumerate(session.criteria)}
    applied = ({item.criterion_id: item.weight for item in weighting.criterion_weights}
               if criterion_weight_override is None else criterion_weight_override)
    values = np.array([applied[cid] for cid in ranking_criterion_ids], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Ranking weights must be finite, nonnegative, with positive retained mass.")
    values /= values.sum()
    ranking_weighting = weighting.model_copy(deep=True)
    ranking_weighting.criterion_weights = [
        item.model_copy(update={"weight": float(value)})
        for cid, value in zip(ranking_criterion_ids, values)
        for item in weighting.criterion_weights if item.criterion_id == cid
    ]
    # This projection is used only by the ranking service; the original session
    # and its full pairwise question/response set remain unchanged.
    ranking_view = session.model_copy(update={
        "criteria": [session.criteria[positions[cid]] for cid in ranking_criterion_ids],
    })
    metadata["ranking_criterion_ids"] = list(ranking_criterion_ids)
    metadata["applied_criterion_weights"] = dict(zip(ranking_criterion_ids, values.tolist()))
    if criterion_weight_override is not None:
        metadata["ranking_weight_source"] = "explicit"
    elif ranking_criterion_ids != list(positions):
        metadata["ranking_weight_source"] = "restricted_and_renormalized"

    ranking = MCDMService.generate_ranking(
        session=ranking_view,
        decision_matrix=decision_matrix[:, [positions[cid] for cid in ranking_criterion_ids]],
        weighting=ranking_weighting,
        config=processing.ranking_method,
    )

    return EntityCalculation(
        entity_id=entity_id,
        scope=scope,
        pairwise_matrix=pairwise_matrix.tolist(),
        weighting=weighting,
        ranking=ranking,
        metadata=metadata,
    )

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
    # experiment =  run.experiment
    # session = experiment.session
    # processing = experiment.processing

    result = _create_run_result(run)

    try:
        result.execution["numerical_status"] = "running"
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
                metadata={
                    "group_id": participant.stakeholder_group_id,
                },
                reused_weighting=resolved.reused_weightings.get(
                    ("participant", participant.id)
                ),
                ranking_criterion_ids=resolved.ranking_criterion_ids,
                criterion_weight_override=resolved.criterion_weight_overrides.get("participant"),
            )

            result.mcdm.participants.append(calculation)
            
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
                metadata={
                    "participant_count": len(members),
                    "voting_power": group.voting_power,
                },
                reused_weighting=resolved.reused_weightings.get(
                    ("group", group.id)
                ),
                ranking_criterion_ids=resolved.ranking_criterion_ids,
                criterion_weight_override=resolved.criterion_weight_overrides.get("group"),
            )

            result.mcdm.groups.append(calculation)

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
        )

        result.set_status("calculated")
        result.execution["numerical_status"] = "completed"

    except Exception as error:
        result.set_status("failed")
        result.execution["numerical_status"] = "failed"

        if result.execution.get("resolution_status") == "running":
            result.execution["resolution_status"] = "failed"

        result.execution["messages"].append(
            f"{type(error).__name__}: {error}"
        )

    return result

def execute_experiments(runs, output, *, ai_config=None, ai_client=None, preflight=None):
    if (ai_config is None) != (ai_client is None):
        raise ValueError(
            "Provide both ai_config and ai_client, or neither."
        )

    ai_enabled = ai_client is not None
    # results = []
    # for run in runs:
    #     # Always execute the numerical stages.
    #     result = execute_run(run)
    #     results.append(result)

    #     if ai_enabled:
    #         # Execute AI stages using trace, ai_config, and ai_client.
    #         ...

    runs = list(runs)
    _validate_execution_order(runs)

    completed = {}
    results = []

    for run in runs:
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

        completed[run.key] = result
        results.append(result)

        # Global experiment ai checker
        # if ai_enabled is False or run.:

    return "", results

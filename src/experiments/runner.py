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
from src.experiments.models import Experiment
from src.experiments.results import RunResult
from src.mcdm.enums import (
    WeightingMethod, RankingMethod, MeanMethod, GroupPowerMethod,
    EmptyGroupPolicy, OnExceedancePolicy, PairwisePreference,
)
from src.mcdm.service import MCDMService
from src.mcdm.models.computation import EntityCalculation
from src.mcdm.reference import DERIVATIONS


ROOT = Path(__file__).resolve().parent.parent

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
    if process.weighting_method.method != WeightingMethod.AHP or process.ranking_method.method != RankingMethod.TOPSIS:
        raise ValueError("Only AHP weighting and TOPSIS ranking are currently supported.")
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
        for field in ("alternative_alignment", "criterion_alignment"):
            if comparison.get(field, "require_identical_ids") != "require_identical_ids":
                raise ValueError(f"Unsupported comparison {field} policy.")
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
) -> EntityCalculation:
    weighting = MCDMService.generate_weights(
        session=session,
        pairwise_matrix=pairwise_matrix,
        config=processing.weighting_method,
    )

    # weights_by_id = {
    #     item.criterion_id: item.weight
    #     for item in weighting.criterion_weights
    # }

    # Validate unique IDs and exact criterion coverage before conversion.
    # weights = np.asarray([
    #     weights_by_id[criterion.id]
    #     for criterion in session.criteria
    # ])

    ranking = MCDMService.generate_ranking(
        session=session,
        decision_matrix=decision_matrix,
        weighting=weighting,
        config=processing.ranking_method,
    )

    return EntityCalculation(
        entity_id=entity_id,
        scope=scope,
        pairwise_matrix=pairwise_matrix.tolist(),
        weighting=weighting,
        ranking=ranking,
        metadata={} if metadata is None else dict(metadata),
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
            for entities in ("criteria", "alternatives"):
                own_ids = {e.id for e in getattr(catalog[key].session, entities)}
                other_ids = {e.id for e in getattr(other.session, entities)}
                if own_ids != other_ids:
                    raise ValueError(
                        f"{key}: comparison {comparison['id']!r} requires identical {entities} IDs."
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

def execute_run(run: PreparedRun):
    experiment =  run.experiment
    session = experiment.session
    processing = experiment.processing

    result = RunResult(
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
            "reference_runs": dict(run.reference_runs),
            "numerical_status": "pending",
            "messages": [],
        },
    )

    try:
        result.execution["numerical_status"] = "running"

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
        )

        result.execution["numerical_status"] = "completed"
        result.status = "calculated"

    except Exception as error:
        result.execution["numerical_status"] = "failed"
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
    results = []
    for run in runs:
        # Always execute the numerical stages.
        result = execute_run(run)
        results.append(result)

        if ai_enabled:
            # Execute AI stages using trace, ai_config, and ai_client.
            ...

    return "", results
        # Store this run's inputs, results, and stage statuses.
# def execute_experiments(
#     *,
#     runs,
#     output,
# ):
#     completed = {}

#     for case in runs:
#         start, tick = timestamp(), time.perf_counter()
#         e = case.experiment

#         print(e)
#         result = RunResult(
#             experiment_id=e.id,
#             variant_id=case.variation.variant_id if case.variation else e.variant_id,
#             case_key=case.key,
#             name=e.name,
#             run_id=uuid4().hex,
#             parent_run_id=completed[case.parent].run_id if case.parent else None, status="failed"
#         )
#         result.execution = {
#             "started_at": start,
#             "messages": [],
#             "description": case.variation.description if case.variation else e.variant_description
#         }

#     try:
#         result.provenance = provenance(e, normalized_inputs(e))
#         if any(completed[d].status != "passed" for d in case.dependencies):
#             result.status = "blocked"
#             result.execution["messages"].append("A required parent or comparison run failed.")
#         else:
#             result.inputs, reused = resolve_inputs(case, completed)
#             result.provenance = provenance(e, result.inputs)
#     except Exception as error:
#         result.status = "failed"
#         result.execution["messages"].append(f"{type(error).__name__}: {error}")

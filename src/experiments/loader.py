"""Load schema-3 experiments and resolve their stakeholder case variations."""
import csv
import json
import math
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from src.mcdm.enums import (
    WeightingMethod, RankingMethod, MeanMethod, GroupPowerMethod,
    EmptyGroupPolicy, OnExceedancePolicy,
)
from src.mcdm.models.domain import (
    Criterion, PairwiseQuestion, CriterionValue, Alternative, StakeholderGroup,
    Participant, PairwiseResponse,
)
from src.mcdm.models.session import DecisionSession
from src.mcdm.models.methods import (
    WeightingMethodConfig, RankingMethodConfig, AggregationMethodConfig,
    ConsistencyMethodConfig,
)
from src.experiments.models import (
    Experiment, ExperimentResources, ExperimentDefaults, ExperimentProcess,
    ExperimentExecution, ExperimentStakeholders, ExperimentAI
)
from src.experiments.participants import _participants, check_keys
from src.experiments.variations import resolve_cases

SUPPORTED_SCHEMA_VERSIONS = {"3.0"}


def _unique(values: Iterable[str], label: str) -> None:
    seen = set()
    for value in values:
        if value in seen:
            raise ValueError(f"Duplicate {label}: {value!r}.")
        seen.add(value)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return data


def _optional_copy(
    mapping: dict[str, Any],
    key: str,
) -> Any:
    if key not in mapping:
        return None

    return deepcopy(mapping[key])


def _ai(data: dict[str, Any]) -> ExperimentAI:
    """Validate and retain AI request settings without invoking a provider."""
    if not isinstance(data, dict):
        raise ValueError("defaults.ai must be an object.")
    check_keys(data, {
        "enabled", "repetitions", "layers", "provider", "model", "parameters",
        "prompt_overrides", "reported_result_interventions",
    }, "defaults.ai")
    enabled = data.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("AI enabled must be bool.")
    repetitions = data.get("repetitions", 1)
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("AI repetitions must be a positive integer.")
    supported_layers = (
        "data_comprehension", "data_explanation_reporting", "policy_decision_advisory",
    )
    layers = data.get("layers", list(supported_layers))
    if (not isinstance(layers, list)
            or any(not isinstance(layer, str) or layer not in supported_layers for layer in layers)
            or len(layers) != len(set(layers)) or (enabled and not layers)):
        raise ValueError("AI layers must be unique supported names, nonempty when AI is enabled.")
    for key in ("provider", "model"):
        value = data.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"AI {key} must be a nonempty string or null.")
    parameters = data.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("AI parameters must be an object.")
    try:
        json.dumps(parameters, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("AI parameters must contain finite JSON values.") from error
    overrides = data.get("prompt_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - set(supported_layers):
        raise ValueError("AI prompt_overrides must map supported layer names to replacements.")
    for replacements in overrides.values():
        if not isinstance(replacements, dict) or set(replacements) - {"system", "task"}:
            raise ValueError("Prompt replacements support only system and task components.")
        if any(not isinstance(value, str) or not value.strip() for value in replacements.values()):
            raise ValueError("Prompt replacements must be nonempty strings.")
    interventions = data.get("reported_result_interventions", [])
    if not isinstance(interventions, list):
        raise ValueError("AI reported_result_interventions must be a list.")
    for intervention in interventions:
        if not isinstance(intervention, dict):
            raise ValueError("Each reported-result intervention must be an object.")
        check_keys(intervention, {
            "type", "scope", "alternative_id", "score", "rank", "rerank_remaining",
        }, "reported-result intervention")
        if intervention.get("type") != "promote_alternative" or intervention.get("scope") != "session":
            raise ValueError("Reported-result interventions support only session alternative promotion.")
        alternative_id = intervention.get("alternative_id")
        if not isinstance(alternative_id, str) or not alternative_id.strip():
            raise ValueError("Reported alternative_id must be a nonempty string.")
        score = intervention.get("score")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Reported score must be finite and between zero and one.")
        if type(intervention.get("rank")) is not int or intervention["rank"] != 1:
            raise ValueError("Alternative promotion requires rank 1.")
        if intervention.get("rerank_remaining") is not True:
            raise ValueError("Alternative promotion requires rerank_remaining=true.")
    # All nested collections are independent from the parsed source document.
    return ExperimentAI(
        enabled=enabled, repetitions=repetitions, layers=deepcopy(layers),
        provider=data.get("provider"), model=data.get("model"),
        parameters=deepcopy(parameters), prompt_overrides=deepcopy(overrides),
        reported_result_interventions=deepcopy(interventions),
    )


def _processing(data: dict[str, Any]) -> ExperimentProcess:
    if not isinstance(data, dict):
        raise ValueError("Processing must be an object.")
    weighting, ranking = data["weighting"], data["ranking"]
    aggregation, consistency = data["aggregation"], data["consistency"]
    if not all(isinstance(item, dict) for item in (weighting, ranking, aggregation, consistency)):
        raise ValueError("Processing sections must be objects.")
    threshold = consistency["threshold"]
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or threshold < 0):
        raise ValueError("Consistency threshold must be finite and nonnegative.")
    empty_policy = aggregation["empty_group_policy"]
    empty_policy = {"exclude": "excluded", "include": "included"}.get(empty_policy, empty_policy)
    return ExperimentProcess(
        weighting_method=WeightingMethodConfig(
            method=WeightingMethod(weighting["method"]),
            parameters=deepcopy(weighting.get("parameters", {})),
        ),
        ranking_method=RankingMethodConfig(
            method=RankingMethod(ranking["method"]),
            parameters=deepcopy(ranking.get("parameters", {})),
        ),
        aggregation_method=AggregationMethodConfig(
            within_group=MeanMethod(aggregation["within_group"]),
            between_groups=MeanMethod(aggregation["between_groups"]),
            group_power_normalization=GroupPowerMethod(aggregation["group_power_normalization"]),
            empty_group_policy=EmptyGroupPolicy(empty_policy),
        ),
        consistency_method=ConsistencyMethodConfig(
            threshold=float(threshold),
            on_exceedance=OnExceedancePolicy(consistency["on_exceedance"]),
        ),
    )


def load_experiment_json(path: Path | str) -> list[Experiment]:
    """Parse one document, generate participants, and construct its resolved cases.

    External comparison references are retained for batch validation. Local input
    inheritance is resolved here; no rankings, AI calls, or assertions execute.
    """
    path = Path(path).resolve()
    data = _read_json(path)
    if data.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("Unsupported experiment schema version.")

    resources = ExperimentResources(**data["resources"])

    # Load criteria
    criteria_data = _read_json(path.parent / resources.criteria)
    criteria = [
        Criterion(**{
            key: item[key]
            for key in ("id", "name", "description", "direction", "unit")
            if key in item
        })
        for item in criteria_data["criteria"]
    ]
    _unique((item.id for item in criteria), "criterion ID")

    # Load questions
    criterion_names = {item.id: item.name for item in criteria}
    questions = []
    for item in criteria_data["pairwise_question_order"]:
        left, right = item["left_criterion_id"], item["right_criterion_id"]
        if left not in criterion_names or right not in criterion_names:
            raise ValueError(
                f"Question {item['question_id']!r} references an unknown criterion."
            )
        questions.append(PairwiseQuestion(
            id=item["question_id"],
            text=f"How important is {criterion_names[left]} compared with {criterion_names[right]}?",
            left_criterion_id=left,
            right_criterion_id=right,
        ))
    _unique((item.id for item in questions), "question ID")

    # Check that every question pair is generated
    pairs = [frozenset((q.left_criterion_id, q.right_criterion_id)) for q in questions]
    if len(set(pairs)) != len(pairs) or set(pairs) != {frozenset(p) for p in combinations(criterion_names, 2)}:
        raise ValueError("Questions must cover every unique criterion pair exactly once.")

    # Load alternatives data
    alternative_data = _read_json(path.parent / resources.alternatives)["alternatives"]
    _unique((item["id"] for item in alternative_data), "alternative ID")

    # Load data
    join_key = resources.dataset_join_key or "alternative_id"
    columns = [item["data_column"] for item in criteria_data["criteria"]]
    with (path.parent / resources.dataset).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = {join_key, *columns} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}.")
        rows = list(reader)
    _unique((row[join_key] for row in rows), "dataset alternative ID")
    rows_by_id = {row[join_key]: row for row in rows}

    # Load alternatives
    alternatives = []
    for item in alternative_data:
        if item["id"] not in rows_by_id:
            raise ValueError(f"Dataset is missing a row for alternative {item['id']!r}.")
        row = rows_by_id[item["id"]]
        values = []
        for criterion, column in zip(criteria, columns):
            try:
                value = float(row[column])
                if not math.isfinite(value):
                    raise ValueError("non-finite value")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Alternative {item['id']!r}, column {column!r}: "
                    f"expected a finite number, got {row[column]!r}."
                ) from error
            values.append(CriterionValue(criterion_id=criterion.id, value=value))
        alternatives.append(Alternative(**item, criterion_values=values))

    # Load Stakeholder Groups
    stakeholder_data = data.get("stakeholders", {})
    groups = [
        StakeholderGroup(**item)
        for item in stakeholder_data.get("stakeholder_groups", [])
    ]
    _unique((item.id for item in groups), "stakeholder group ID")


    # Parse predefined participants, then append generator results.
    if stakeholder_data.get("judgment_encoding") != "pairwise_preference":
        raise ValueError("Schema 3.0 requires judgment_encoding=pairwise_preference.")
    config = stakeholder_data["participants"]
    if not isinstance(config, dict):
        raise ValueError("stakeholders.participants must be an object.")
    check_keys(config, {"predefined", "generators", "combine", "on_id_collision"}, "participants")
    if config.get("combine", "append") != "append" or config.get("on_id_collision", "error") != "error":
        raise ValueError("Only combine=append and on_id_collision=error are supported.")
    participants = _participants(
        predefined=config.get("predefined", []),
        generators=config.get("generators", []),
        label=data["experiment_id"], schema_version=data["schema_version"],
        group_ids={group.id for group in groups},
        question_ids={question.id for question in questions},
        judgment_encoding=stakeholder_data["judgment_encoding"],
    )

    raw_defaults = data["defaults"]
    repetitions = raw_defaults["repetitions"]
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer.")
    seed = raw_defaults["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    process = _processing(raw_defaults["processing"])
    ai = _ai(raw_defaults["ai"])
    pipeline = raw_defaults["pipeline"]
    if pipeline not in ("mcdm", "ai-mcdm"):
        raise ValueError("Unsupported experiment pipeline.")
    if (pipeline == "ai-mcdm") != ai.enabled:
        raise ValueError("ai-mcdm requires AI enabled; mcdm requires AI disabled.")
    output = data.get("output", {})
    if not isinstance(output, dict):
        raise ValueError("output must be an object.")
    for key in ("save_ai_calls", "save_ai_responses"):
        if key in output and not isinstance(output[key], bool):
            raise ValueError(f"output.{key} must be bool.")
    defaults = ExperimentDefaults(
        pipeline=pipeline, seed=seed, repetition=repetitions,
        processing=process, ai=deepcopy(ai),
    )
    resolved_cases = resolve_cases(
        {path: data}, base_inputs={data["experiment_id"]: (participants, groups)},
        validate_references=False,
    )

    scenario = data["scenario"]
    experiments = []
    for resolved in resolved_cases:
        case = resolved.case
        session = DecisionSession(
            id=resolved.key, title=case["name"], description=case["description"],
            criteria=deepcopy(criteria), questions=deepcopy(questions),
            alternatives=deepcopy(alternatives), stakeholder_groups=deepcopy(resolved.groups),
            participants=[Participant(
                id=p.participant_id, name=p.participant_id,
                stakeholder_group_id=p.group_id,
                responses=[PairwiseResponse(question_id=qid, preference=value)
                           for qid, value in p.judgments.items()],
            ) for p in resolved.participants],
        )
        experiments.append(Experiment(
            schema_version=data["schema_version"],
            id=data["experiment_id"],
            name=data["name"],
            research_description=data["description"],
            research_purpose=data["purpose"],
            scenario_id=data["scenario"]["id"],
            scenario_description=data["scenario"]["description"],
            scenario_policy_question=data["scenario"]["policy_question"],
            scenario_context=_optional_copy(
                scenario,
                "context",
            ),
            scenario_objectives=_optional_copy(
                scenario,
                "objectives",
            ),
            scenario_assumptions=_optional_copy(
                scenario,
                "assumptions",
            ),
            scenario_constraints=_optional_copy(
                scenario,
                "constraints",
            ),
            resources=deepcopy(resources),
            processing=deepcopy(process),
            execution=ExperimentExecution(
                random_seed=seed, consistency_threshold=process.consistency_method.threshold,
            ),
            ai=deepcopy(ai),
            session=session,
            source_path=str(path),
            variant_id=resolved.key,
            variant_description=case["description"],
            stakeholders=ExperimentStakeholders(
                stakeholder_groups=deepcopy(resolved.groups),
                participants=deepcopy(resolved.participants),
            ),
            expected_assertions=resolved.assertions,
            output=deepcopy(output),
            defaults=deepcopy(defaults),
            case_metadata=deepcopy(case),
            baseline_case=data["baseline"]["case"],
            comparisons=deepcopy(case.get("comparisons", [])),
            judgment_encoding=stakeholder_data["judgment_encoding"],
            scale_conversion=deepcopy(stakeholder_data.get("scale_conversion", {})),
        ))
    return experiments


def load_experiments(experiments_dir: Path | str | None = None) -> list[Experiment]:
    directory = (Path(experiments_dir) if experiments_dir is not None
                 else Path(__file__).resolve().parents[2] / "experiments")
    if not directory.is_dir():
        raise ValueError(f"Experiment directory does not exist: {directory}")
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError("No experiment JSON files found.")
    experiments = []
    source_ids = []
    for path in paths:
        try:
            cases = load_experiment_json(path)
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise ValueError(f"{path}: {error}") from error
        source_ids.append(cases[0].id)
        experiments.extend(cases)
    _unique(source_ids, "experiment IDs")
    _unique((e.session.id for e in experiments), "case IDs")
    available = {e.session.id for e in experiments}
    for experiment in experiments:
        references = [experiment.baseline_case]
        references += [c["reference"] for c in experiment.comparisons]
        references += [a["parameters"]["reference"] for a in experiment.expected_assertions
                       if "reference" in a.get("parameters", {})]
        for reference in references:
            if reference not in available:
                raise ValueError(f"{experiment.session.id}: unknown reference {reference!r}.")
    return experiments

from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from src.variations import ExecutionVariation
from src.enum import PairwisePreference, RankingMethod, WeightingMethod
from src.experiment import (
    Experiment,
    ExperimentExecution,
    ExperimentParticipant,
    ExperimentProcess,
    ExperimentResources,
    ExperimentStakeholders,
    ParticipantVariation,
)
from src.models.domain import (
    Alternative,
    Criterion,
    CriterionValue,
    PairwiseQuestion,
    PairwiseResponse,
    Participant,
    StakeholderGroup,
)
from src.models.methods import RankingMethodConfig, WeightingMethodConfig
from src.models.session import DecisionSession


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


def _participants(items: list[dict[str, Any]], label: str, schema_version: str = "1.0") -> list[ExperimentParticipant]:
    if not isinstance(items, list):
        raise ValueError("Participants must be a list.")
    participants = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("judgments"), dict):
            raise ValueError("Each participant requires a judgments object.")
        judgments = {}
        for question_id, value in item["judgments"].items():
            if schema_version == "2.0":
                from src.reference import validate_ratio
                validate_ratio(value)
                judgments[question_id] = float(value)
                continue
            # The stored codes are -4..4, not the resulting Saaty ratios.
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise ValueError(
                    f"Participant {item['participant_id']!r}, question {question_id!r}: "
                    f"invalid judgment {value!r}; expected a code from -4 to 4."
                )
            try:
                judgments[question_id] = PairwisePreference(str(value))
            except ValueError as error:
                raise ValueError(
                    f"Participant {item['participant_id']!r}, question {question_id!r}: "
                    f"invalid judgment {value!r}; expected a code from -4 to 4."
                ) from error
        if not judgments:
            raise ValueError(f"Participant {item['participant_id']!r} has no judgments.")
        participants.append(ExperimentParticipant(
            participant_id=item["participant_id"],
            group_id=item["group_id"],
            judgments=judgments,
        ))
    _unique((item.participant_id for item in participants), f"participant ID in {label}")
    return participants


def _processing(data: dict[str, Any]) -> ExperimentProcess:
    if not isinstance(data, dict):
        raise ValueError("Processing must be an object.")
    if "weighting_method" in data and "weight_method" in data and data["weighting_method"] != data["weight_method"]:
        raise ValueError("Conflicting weighting method locations.")
    weighting = data.get("weighting_method", data.get("weight_method"))
    ranking = data["ranking_method"]
    for config, legacy in ((weighting, "weighting"), (ranking, "ranking")):
        if isinstance(config, dict) and legacy in data and config.get("parameters", {}) != data[legacy]:
            raise ValueError(f"Conflicting {legacy} parameter locations.")
    if not isinstance(weighting, dict):
        weighting = {"method": weighting, "parameters": data.get("weighting", {})}
    if not isinstance(ranking, dict):
        ranking = {"method": ranking, "parameters": data.get("ranking", {})}
    for config in (weighting, ranking):
        if not isinstance(config.get("parameters", {}), dict):
            raise ValueError("Method parameters must be an object.")
    return ExperimentProcess(
        weighting_method=WeightingMethodConfig(
            method=WeightingMethod(weighting["method"]),
            parameters=deepcopy(weighting.get("parameters", {})),
        ),
        ranking_method=RankingMethodConfig(
            method=RankingMethod(ranking["method"]),
            parameters=deepcopy(ranking.get("parameters", {})),
        ),
    )


def _load_experiment(path: Path) -> list[Experiment]:
    data = _read_json(path)
    if data.get("schema_version") not in {"1.0", "2.0"}:
        raise ValueError("Unsupported experiment schema version.")
    resources = ExperimentResources(**data["resources"])
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
    from itertools import combinations
    pairs = [frozenset((q.left_criterion_id, q.right_criterion_id)) for q in questions]
    if len(set(pairs)) != len(pairs) or set(pairs) != {frozenset(p) for p in combinations(criterion_names, 2)}:
        raise ValueError("Questions must cover every unique criterion pair exactly once.")

    alternative_data = _read_json(path.parent / resources.alternatives)["alternatives"]
    _unique((item["id"] for item in alternative_data), "alternative ID")
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

    stakeholder_data = data.get("stakeholders", {})
    groups = [
        StakeholderGroup(**item)
        for item in stakeholder_data.get("stakeholder_groups", [])
    ]
    _unique((item.id for item in groups), "stakeholder group ID")
    base = _participants(
        stakeholder_data.get("participant", stakeholder_data.get("participants", [])),
        "base participants", data["schema_version"],
    )
    variants = [
        ParticipantVariation(
            variant_id=item["variant_id"],
            description=item["description"],
            participants=_participants(item["participants"], item["variant_id"], data["schema_version"]),
        )
        for item in stakeholder_data.get("participant_variants", [])
    ]
    _unique((item.variant_id for item in variants), "participant variant ID")
    known_groups = {item.id for item in groups}
    known_questions = {item.id for item in questions}
    for participants in [base, *(item.participants for item in variants)]:
        for participant in participants:
            if participant.group_id not in known_groups:
                raise ValueError(
                    f"Participant {participant.participant_id!r} references unknown "
                    f"stakeholder group {participant.group_id!r}."
                )
            missing = known_questions - participant.judgments.keys()
            if missing:
                raise ValueError(f"Participant {participant.participant_id!r} is missing judgments: {sorted(missing)}.")
            unknown = participant.judgments.keys() - known_questions
            if unknown:
                raise ValueError(
                    f"Participant {participant.participant_id!r} references "
                    f"unknown questions: {sorted(unknown)}."
                )
    stakeholders = ExperimentStakeholders(groups, base, variants)
    execution_data = deepcopy(data["execution"])
    execution_data["variations"] = [ExecutionVariation(**v) for v in execution_data.get("variations", [])]
    execution = ExperimentExecution(**execution_data)
    if execution.variants and execution.variations:
        raise ValueError("Use participant variants or execution variations, not both.")
    if isinstance(execution.consistency_threshold, bool) or not math.isfinite(execution.consistency_threshold) or execution.consistency_threshold < 0:
        raise ValueError("Consistency threshold must be finite and nonnegative.")
    if isinstance(execution.random_seed, bool) or not isinstance(execution.random_seed, int):
        raise ValueError("random_seed must be an integer.")
    if not isinstance(execution.variants, bool):
        raise ValueError("execution.variants must be a boolean.")
    if execution.variants and (
        not variants or any(not item.participants for item in variants)
    ):
        raise ValueError("Enabled variants require nonempty participant sets.")
    if not execution.variants and not base:
        raise ValueError("Base execution requires a nonempty participant set.")

    experiment_id = data["experiment_id"] if "experiment_id" in data else data["id"]
    scenario = data.get("scenario", {})
    scenario_id = data.get("scenario_id", scenario.get("id"))
    scenario_description = data.get("scenario_description", scenario.get("description"))
    scenario_policy_question = data.get("scenario_policy_question", scenario.get("policy_question"))
    if scenario_id is None or scenario_description is None:
        raise ValueError("Scenario ID and description are required.")
    processing = _processing(data["processing"])
    runs = (
        [(item.variant_id, item.description, item.participants) for item in variants]
        if execution.variants else [(None, None, base)]
    )
    experiments = []
    for variant_id, variant_description, participants in runs:
        session = DecisionSession(
            id=variant_id if variant_id is not None else experiment_id,
            title=data["name"],
            description=scenario_description,
            criteria=deepcopy(criteria),
            stakeholder_groups=deepcopy(groups),
            questions=deepcopy(questions),
            participants=[
                Participant(
                    id=item.participant_id,
                    name=item.participant_id,
                    stakeholder_group_id=item.group_id,
                    responses=[
                        PairwiseResponse(question_id=key, preference=value)
                        for key, value in item.judgments.items()
                    ],
                )
                for item in participants
            ],
            alternatives=deepcopy(alternatives),
        )
        experiments.append(Experiment(
            source_path=str(path.resolve()),
            schema_version=data["schema_version"],
            id=experiment_id,
            name=data["name"],
            research_purpose=data["research_purpose"],
            scenario_id=scenario_id,
            scenario_description=scenario_description,
            scenario_policy_question=scenario_policy_question,
            resources=deepcopy(resources),
            processing=deepcopy(processing),
            execution=deepcopy(execution),
            session=session,
            variant_id=variant_id,
            variant_description=variant_description,
            stakeholders=deepcopy(stakeholders),
            expected_assertions=deepcopy(data.get("expected_assertions", [])),
            ai_evaluation=deepcopy(data.get("ai_evaluation", {})),
            output=deepcopy(data.get("output", {})),
        ))
    return experiments


def load_experiments(experiments_dir: Path | str | None = None) -> list[Experiment]:
    """Load sorted experiment files, expanding enabled variants into independent runs.

    Resource paths are relative to their experiment file, independent of the cwd.
    """
    directory = (
        Path(experiments_dir)
        if experiments_dir is not None
        else Path(__file__).resolve().parent.parent / "experiments"
    )
    if not directory.is_dir():
        raise ValueError(f"Experiments directory does not exist: {directory}")
    experiments = []
    source_ids = []
    for path in sorted(directory.glob("*.json")):
        try:
            runs = _load_experiment(path)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}: {error}") from error
        source_ids.append(runs[0].id)
        experiments.extend(runs)
    if not experiments:
        raise ValueError("No experiment JSON files found.")
    _unique(source_ids, "experiment ID")
    _unique((item.session.id for item in experiments), "session ID")
    return experiments

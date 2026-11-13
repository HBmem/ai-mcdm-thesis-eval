import csv
import json
import math

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable
from itertools import combinations

from src.mcdm.enums import WeightingMethod, RankingMethod, MeanMethod, GroupPowerMethod, EmptyGroupPolicy, PairwisePreference
from src.mcdm.models.domain import Criterion, PairwiseQuestion, CriterionValue, Alternative, StakeholderGroup
from src.mcdm.models.methods import WeightingMethodConfig, RankingMethodConfig, AggregationMethodConfig, ConsistencyMethodConfig
from src.experiments.models import Experiment, ExperimentResources, ExperimentDefaults, ExperimentProcess, AIConfig, ExperimentParticipant, ParticipantVariation

# Version 1.0 and 2.0 are no longer supported
SUPPORTED_SCHEMA_VERSIONS = {"3.0"}
SUPPORTED_GENERATOR_STRATEGIES = ["Repeat"]

def _unique(values: Iterable[str], label: str) -> None:
    """"""

    seen = set()
    for value in values:
        if value in seen:
            raise ValueError(f"Duplicate {label}: {value!r}.")
        seen.add(value)


def _read_json(path: Path) -> dict[str, Any]:
    """"""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return data

# def _participants(
#     items: list[dict[str, Any]],
#     label: str, schema_version: str = "1.0"
# ) -> list[ExperimentParticipant]:
#     if not isinstance(items, list):
#         raise ValueError("Participants must be a list.")

#     participants = []
#     for item in items:
#         if not isinstance(item, dict) or not isinstance(item.get("judgments"), dict):
#             raise ValueError("Each participant requires a judgments object.")

# def _load_experiment(path: Path) -> list[Experiment]:
#     """"""

#     data = _read_json(path)
#     if data.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
#         raise ValueError("Unsupported experiment schema version.")
    
#     resources = ExperimentResources(**data["resources"])

#     # Criteria
#     criteria_data = _read_json(path.parent / resources.criteria)
#     criteria_data = _read_json(path.parent / resources.criteria)
#     criteria = [
#         Criterion(**{
#             key: item[key]
#             for key in ("id", "name", "description", "direction", "unit")
#             if key in item
#         })
#         for item in criteria_data["criteria"]
#     ]
#     _unique((item.id for item in criteria), "criterion ID")
#     criterion_names = {item.id: item.name for item in criteria}

#     # Questions
#     questions = []
#     for item in criteria_data["pairwise_question_order"]:
#         left, right = item["left_criterion_id"], item["right_criterion_id"]
#         if left not in criterion_names or right not in criterion_names:
#             raise ValueError(
#                 f"Question {item['question_id']!r} references an unknown criterion."
#             )
#         questions.append(PairwiseQuestion(
#             id=item["question_id"],
#             text=f"How important is {criterion_names[left]} compared with {criterion_names[right]}?",
#             left_criterion_id=left,
#             right_criterion_id=right,
#         ))
#     _unique((item.id for item in questions), "question ID")

#     pairs = [frozenset((q.left_criterion_id, q.right_criterion_id)) for q in questions]
#     if len(set(pairs)) != len(pairs) or set(pairs) != {frozenset(p) for p in combinations(criterion_names, 2)}:
#         raise ValueError("Questions must cover every unique criterion pair exactly once.")

#     # Alternatives
#     alternative_data = _read_json(path.parent / resources.alternatives)["alternatives"]
#     _unique((item["id"] for item in alternative_data), "alternative ID")
#     join_key = resources.dataset_join_key or "alternative_id"
#     columns = [item["data_column"] for item in criteria_data["criteria"]]
#     with (path.parent / resources.dataset).open(encoding="utf-8-sig", newline="") as stream:
#         reader = csv.DictReader(stream)
#         missing = {join_key, *columns} - set(reader.fieldnames or [])
#         if missing:
#             raise ValueError(f"Dataset is missing columns: {sorted(missing)}.")
#         rows = list(reader)
#     _unique((row[join_key] for row in rows), "dataset alternative ID")
#     rows_by_id = {row[join_key]: row for row in rows}
#     alternatives = []
#     for item in alternative_data:
#         if item["id"] not in rows_by_id:
#             raise ValueError(f"Dataset is missing a row for alternative {item['id']!r}.")
#         row = rows_by_id[item["id"]]
#         values = []
#         for criterion, column in zip(criteria, columns):
#             try:
#                 value = float(row[column])
#                 if not math.isfinite(value):
#                     raise ValueError("non-finite value")
#             except (TypeError, ValueError) as error:
#                 raise ValueError(
#                     f"Alternative {item['id']!r}, column {column!r}: "
#                     f"expected a finite number, got {row[column]!r}."
#                 ) from error
#             values.append(CriterionValue(criterion_id=criterion.id, value=value))
#         alternatives.append(Alternative(**item, criterion_values=values))

#     # Participants
#     stakeholder_data = data.get("stakeholders", {})
#     groups = [
#         StakeholderGroup(**item)
#         for item in stakeholder_data.get("stakeholder_groups", [])
#     ]
#     _unique((item.id for item in groups), "stakeholder group ID")
#     base = _participants(
#         stakeholder_data.get("participant", stakeholder_data.get("participants", [])),
#         "base participants", data["schema_version"],
#     )
#     variants = [
#         ParticipantVariation(
#             variant_id=item["variant_id"],
#             description=item["description"],
#             participants=_participants(item["participants"], item["variant_id"], data["schema_version"]),
#         )
#         for item in stakeholder_data.get("participant_variants", [])
#     ]
#     _unique((item.variant_id for item in variants), "participant variant ID")

# def load_experiments(experiments_dir: Path | str | None = None) -> list[Experiment]:
#     """
#     Load sorted experiment files, expanding enabled variants into independent runs.

#     Resource paths are relative to their experiment file, independent of the cwd.
#     """

#     directory = (
#         Path(experiments_dir)
#         if experiments_dir is not None
#         else Path(__file__).resolve().parent.parent / "experiments"
#     )

#     if not directory.is_dir():
#         raise ValueError(f"Experiment directory does not exists: {directory}")

#     experiments = []
#     source_ids = []

#     for path in sorted(directory.glob("*.json")):
#         try:
#             runs = _load_experiment(path)
#         except (ValueError, KeyError, TypeError) as e:
#             raise ValueError(f"{path}: {e}") from e
#         source_ids.append(runs[0].id)
#         experiments.extend(runs)
#     if not experiments:
#         raise ValueError("No experiment JSON files found.")
    
#     _unique(source_ids, "experiment ID")
#     _unique((item.session.id for item in experiments), "session ID")
#     return experiments

def _participants(
    predefined: list[dict[str, Any]],
    generators: list[dict[str, Any]],
    label: str,
    schema_version: str = "3.0"
) -> list[ExperimentParticipant]:
    if not isinstance(predefined, list):
        raise ValueError("Predefined participants must be a list.")
    if not isinstance(generators, list):
        raise ValueError("Generated participant rules instructions must be a list.")

    participants = []
    for item in predefined:
        if not isinstance(item, dict) or not isinstance(item.get("judgments"), dict):
            raise ValueError("Each participant requires a judgments object.")
        judgments = {}
        for question_id, value in item["judgments"].items():
            if schema_version != "3.0":
                raise ValueError("No configuration for this schema version.")

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

    for item in generators:
        if not isinstance(item, dict):
            raise ValueError("Each participant requires a judgments object.")
        
        # TODO: Add code for generators
    _unique((item.participant_id for item in participants), f"participant ID in {label}")

def _processing(data: dict[str, Any]) -> ExperimentProcess:
    if not isinstance(data, dict):
        raise ValueError("Processing must be an object.")
    
    weighting = data["weighting"]
    ranking = data["ranking"]
    aggregation = data["aggregation"]
    consistency = data["consistency"]

    if not isinstance(weighting, dict):
        weighting = {"method": weighting, "parameters": data.get("weighting", {})}
    if not isinstance(ranking, dict):
        ranking = {"method": ranking, "parameters": data.get("ranking", {})}

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
            empty_group_policy=EmptyGroupPolicy(aggregation["empty_group_policy"])
        ),
        consistency_method=ConsistencyMethodConfig(
            threshold=consistency["threshold"],
            on_exceedance=consistency["on_exceedance"]
        )
    )

def load_experiment_json(path: Path) -> list[Experiment]:
    data = _read_json(path)

    if data.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("Unsupported experiment schema version.")

    resources = ExperimentResources(**data["resources"])
    print("Resources", resources)

    # # Load experiment defaults
    # default_data = data.get("defaults", {})
    # default_processing = ExperimentProcess(
    #     weighting_method=WeightingMethodConfig(
    #         method=
    #     )
    # )
    # defaults = ExperimentDefaults(
    #     pipeline=default_data["pipeline"],
    #     seed=default_data["seed"],
    #     repetition=default_data["repetition"],
    #     processing=default_processing,
    #     ai=AIConfig(
    #         enabled=default_processing["ai"]
    #     )
    # )

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
    print("Criteria", criteria)

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
    print("Questions", questions)

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
    print("Alternatives", alternatives)

    # Load Stakeholder Groups
    stakeholder_data = data.get("stakeholders", {})
    groups = [
        StakeholderGroup(**item)
        for item in stakeholder_data.get("stakeholder_groups", [])
    ]
    _unique((item.id for item in groups), "stakeholder group ID")
    
    participants = _participants(
        stakeholder_data.get("participants")
    )

def load_experiments(experiments_dir: Path | str | None = None) -> list[Experiment]:
    """"""

    directory = (
        Path(experiments_dir)
        if experiments_dir is not None
        else Path(__file__).resolve().parent.parent / "experiments"
    )

    if not directory.is_dir():
        raise ValueError(f"Experiment directory does not exists: {directory}")

    experiments = []

    for path in sorted(directory.glob("*.json")):
        try:
            # runs = load_experiment_json(path)
            load_experiment_json(path)
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"{path}: {e}") from e
        print("test")
        # experiments.extend(runs)
    if not experiments:
        raise ValueError("No experiment JSON files found.")

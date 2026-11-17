"""Resolve schema-3 stakeholder operations without calculating MCDM results."""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from src.experiments.models import ExperimentParticipant
from src.experiments.participants import (
    _participants, unique as _unique, check_keys as _keys, resize_group,
    parse_preference, validate_participants,
)
from src.mcdm.models.domain import StakeholderGroup


@dataclass
class ResolvedCase:
    key: str
    source_path: Path
    document: dict[str, Any]
    case: dict[str, Any]
    participants: list[ExperimentParticipant]
    groups: list[StakeholderGroup]

    @property
    def assertions(self):
        return deepcopy(self.document.get("assertions", []) + self.case.get("assertions", []))


def resolve_cases(documents, *, base_inputs=None, validate_references=True):
    """Resolve stakeholder variations without performing calculations.

    documents maps JSON file paths to their parsed objects. Register every case
    before resolving inputs, so file/case order does not affect inheritance.
    Input inheritance is local; stakeholder operations run in declaration order.
    Result-dependent runtime_operations remain the executor's responsibility.
    Comparisons and assertion references may cross experiment boundaries.
    """
    registry, bases, question_ids = {}, {}, {}
    experiment_ids = set()
    for source, document in documents.items():
        source = Path(source).resolve()
        if document.get("schema_version") != "3.0":
            raise ValueError(f"{source}: expected schema_version 3.0.")
        eid = document["experiment_id"]
        if not isinstance(eid, str) or not eid or "/" in eid or eid in experiment_ids:
            raise ValueError("Experiment IDs must be unique nonempty names without '/'.")
        experiment_ids.add(eid)
        criteria_path = source.parent / document["resources"]["criteria"]
        criteria = json.loads(criteria_path.read_text(encoding="utf-8"))
        question_ids[eid] = {
            q["question_id"] for q in criteria["pairwise_question_order"]
        }
        if base_inputs is not None:
            participants, groups = deepcopy(base_inputs[eid])
        else:
            stakeholders = document["stakeholders"]
            if stakeholders.get("judgment_encoding") != "pairwise_preference":
                raise ValueError("Only pairwise_preference judgment encoding is supported.")
            groups = [StakeholderGroup(**g) for g in stakeholders["stakeholder_groups"]]
            _unique((g.id for g in groups), "group IDs")
            config = stakeholders["participants"]
            _keys(config, {"predefined", "generators", "combine", "on_id_collision"},
                  "participant configuration")
            if config.get("combine", "append") != "append":
                raise ValueError("Only participant combine=append is implemented.")
            if config.get("on_id_collision", "error") != "error":
                raise ValueError("Only participant on_id_collision=error is implemented.")
            participants = _participants(
                config.get("predefined", []), config.get("generators", []), eid,
                group_ids={g.id for g in groups},
                question_ids=question_ids[eid],
            )
        bases[eid] = (participants, groups)
        if not isinstance(document["cases"], list) or not document["cases"]:
            raise ValueError("An experiment requires at least one case.")
        for case in document["cases"]:
            cid = case["case_id"]
            if not isinstance(cid, str) or not cid or "/" in cid:
                raise ValueError("case_id must be a nonempty name without '/'.")
            key = f"{eid}/{cid}"
            if key in registry:
                raise ValueError(f"Duplicate case {key!r}.")
            registry[key] = (source, document, case)

    # Comparison edges are references, not input-inheritance edges. Bidirectional
    # comparisons therefore do not create input cycles.
    for key, (_, document, case) in registry.items():
        references = [document["baseline"]["case"]]
        references += [c["reference"] for c in case.get("comparisons", [])]
        for assertion in document.get("assertions", []) + case.get("assertions", []):
            if "reference" in assertion.get("parameters", {}):
                references.append(assertion["parameters"]["reference"])
        for reference in references:
            if (validate_references or reference.split("/", 1)[0] == document["experiment_id"]) and reference not in registry:
                raise ValueError(f"{key}: unknown reference {reference!r}.")

    resolved, visiting = {}, set()

    def visit(key):
        if key in resolved:
            return resolved[key]
        if key in visiting:
            raise ValueError(f"Input inheritance cycle at {key!r}.")
        if key not in registry:
            raise ValueError(f"Unknown input_from case {key!r}.")
        visiting.add(key)
        source, document, case = registry[key]
        eid = document["experiment_id"]
        parent_key = case.get("input_from")
        if parent_key:
            if parent_key.split("/", 1)[0] != eid:
                raise ValueError("input_from is currently supported within one experiment only.")
            parent = visit(parent_key)
            participants, groups = deepcopy((parent.participants, parent.groups))
        else:
            participants, groups = deepcopy(bases[eid])
        operations = case.get("operations", [])
        if not isinstance(operations, list):
            raise ValueError("Case operations must be a list.")
        for operation in operations:
            if not isinstance(operation, dict) or not isinstance(operation.get("type"), str):
                raise ValueError("Each case operation must be an object with a string type.")
            if operation["type"] == "resize_group":
                if operation["group_id"] not in {g.id for g in groups}:
                    raise ValueError("resize_group references an unknown group.")
                participants = resize_group(participants, operation)
            elif operation["type"] == "set_group_voting_powers":
                _keys(operation, {"type", "voting_powers"}, "set_group_voting_powers")
                powers = operation["voting_powers"]
                if not isinstance(powers, dict) or set(powers) - {g.id for g in groups}:
                    raise ValueError("voting_powers must map known group IDs to numbers.")
                for value in powers.values():
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or not 0 < value <= 1):
                        raise ValueError("Voting powers must be finite numbers in (0, 1].")
                # model_copy(update=...) would bypass validation of new powers.
                groups = [StakeholderGroup.model_validate({
                    **g.model_dump(), "voting_power": powers.get(g.id, g.voting_power)
                }) for g in groups]
            elif operation["type"] in {"set_group_judgments", "set_participant_judgments"}:
                kind = operation["type"]
                _keys(operation, {"type", "judgments"}, kind)
                judgments = operation.get("judgments")
                by_group = kind == "set_group_judgments"
                target_ids = ({g.id for g in groups} if by_group
                              else {p.participant_id for p in participants})
                if not isinstance(judgments, dict) or set(judgments) - target_ids:
                    raise ValueError(f"{kind}: judgments must map known target IDs to responses.")
                replacements = {}
                for target, responses in judgments.items():
                    if not isinstance(responses, dict) or set(responses) != question_ids[eid]:
                        raise ValueError(f"{kind}: judgments must cover every question exactly once.")
                    replacements[target] = {
                        qid: parse_preference(value) for qid, value in responses.items()
                    }
                for participant in participants:
                    target = participant.group_id if by_group else participant.participant_id
                    if target in replacements:
                        participant.judgments = deepcopy(replacements[target])
            else:
                raise ValueError(f"Unsupported case operation {operation['type']!r}.")
        if not participants:
            raise ValueError(f"{key}: at least one participant is required.")
        _unique((p.participant_id for p in participants), "resolved participant IDs")
        validate_participants(
            participants, group_ids={g.id for g in groups}, question_ids=question_ids[eid],
        )
        result = ResolvedCase(key, source, deepcopy(document), deepcopy(case), participants, groups)
        _unique((a["id"] for a in result.assertions), f"assertion IDs in {key}")
        resolved[key] = result
        visiting.remove(key)
        return result

    for key in registry:
        visit(key)
    return list(resolved.values())

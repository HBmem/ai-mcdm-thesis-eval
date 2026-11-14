"""Preference-code parsing and extensible participant generation."""
from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any

from src.experiments.models import ExperimentParticipant
from src.mcdm.enums import PairwisePreference


def unique(values: Iterable[str], label: str) -> None:
    values = list(values)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}.")


def nonnegative_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer.")
    return value


def check_keys(data: dict, allowed: set[str], label: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unsupported {label} fields: {sorted(unknown)}")


def parse_preference(value: Any) -> PairwisePreference:
    """Decode legacy codes, preserving the enum until the calculation boundary.

    JSON integers and canonical code strings are accepted; floats and booleans
    are rejected so ratios cannot be silently mistaken for encoded preferences.
    Change scale conversion in PairwisePreference.ahp_scale, not in generators.
    """
    if isinstance(value, PairwisePreference):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"Invalid preference {value!r}; expected a code from -4 to 4.")
    try:
        return PairwisePreference(str(value))
    except ValueError as error:
        raise ValueError(f"Invalid preference {value!r}; expected a code from -4 to 4.") from error


def validate_participants(participants, *, group_ids, question_ids):
    """Validate both parsed data and the output of registered strategies."""
    unique((p.participant_id for p in participants), "participant IDs")
    for p in participants:
        if not isinstance(p.participant_id, str) or not p.participant_id:
            raise ValueError("participant_id must be a nonempty string.")
        if p.group_id not in group_ids:
            raise ValueError(f"{p.participant_id}: unknown group {p.group_id!r}.")
        if not isinstance(p.judgments, dict) or set(p.judgments) != set(question_ids):
            raise ValueError(f"{p.participant_id}: judgments must cover every question exactly once.")
        if not all(isinstance(v, PairwisePreference) for v in p.judgments.values()):
            raise ValueError("Strategies must return parsed PairwisePreference judgments.")


GeneratorStrategy = Callable[
    [dict[str, Any], list[ExperimentParticipant]], list[ExperimentParticipant]
]
_GENERATOR_STRATEGIES: dict[str, GeneratorStrategy] = {}


def register_generator_strategy(name: str):
    """Register a rule handler; duplicate names fail instead of replacing code."""
    if not isinstance(name, str) or not name:
        raise ValueError("A generator strategy requires a nonempty name.")

    def register(handler: GeneratorStrategy):
        if name in _GENERATOR_STRATEGIES:
            raise ValueError(f"Generator strategy {name!r} is already registered.")
        _GENERATOR_STRATEGIES[name] = handler
        return handler
    return register


def generator_strategies() -> dict[str, GeneratorStrategy]:
    """Return a snapshot of available strategies for dispatch or inspection."""
    return dict(_GENERATOR_STRATEGIES)


def format_participant_id(pattern, group_id, index, generator_id=""):
    if not isinstance(pattern, str):
        raise ValueError("id_pattern must be a string.")
    try:
        pid = pattern.format(group_id=group_id, index=index, generator_id=generator_id)
    except (KeyError, ValueError, IndexError, AttributeError) as error:
        raise ValueError(f"Invalid participant id_pattern: {pattern!r}") from error
    if not pid:
        raise ValueError("Generated participant ID cannot be empty.")
    return pid


def select_templates(ids, by_id, group_id):
    if not isinstance(ids, list) or not ids:
        raise ValueError("template_participant_ids must be a nonempty list.")
    unique(ids, "template participant IDs")
    selected = []
    for pid in ids:
        if pid not in by_id or by_id[pid].group_id != group_id:
            raise ValueError(f"Template {pid!r} must exist in group {group_id!r}.")
        selected.append(by_id[pid])
    return selected


@register_generator_strategy("repeat_templates")
def repeat_templates(rule, predefined):
    check_keys(rule, {"generator_id", "strategy", "group_id", "count",
                      "template_participant_ids", "id_pattern"}, "repeat_templates")
    gid = rule["group_id"]
    selected = select_templates(rule["template_participant_ids"],
                                {p.participant_id: p for p in predefined}, gid)
    pattern = rule.get("id_pattern", "{group_id}_{generator_id}_{index:04d}")
    return [
        ExperimentParticipant(
            participant_id=format_participant_id(pattern, gid, i + 1, rule["generator_id"]),
            group_id=gid,
            judgments=deepcopy(selected[i % len(selected)].judgments),
        )
        for i in range(rule["count"])
    ]


def generate_participants(rules, predefined, *, group_ids, question_ids):
    """Return only new participants. Each rule sees independent base templates.

    count means additional participants. Strategies cannot mutate caller inputs
    or use outputs of earlier rules as templates. Collisions are always errors.
    """
    if not isinstance(rules, list):
        raise ValueError("generators must be a list.")
    generated = []
    rule_ids = set()
    strategies = generator_strategies()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("Each generator must be an object.")
        rid = rule.get("generator_id")
        if not isinstance(rid, str) or not rid or rid in rule_ids:
            raise ValueError("Generator IDs must be unique nonempty strings.")
        rule_ids.add(rid)
        if rule.get("group_id") not in group_ids:
            raise ValueError(f"Unknown generator group {rule.get('group_id')!r}.")
        count = nonnegative_count(rule.get("count"), "Generator count")
        name = rule.get("strategy")
        if not isinstance(name, str) or name not in strategies:
            raise ValueError(f"Unsupported generator strategy {name!r}; choose from {sorted(strategies)}.")
        batch = strategies[name](deepcopy(rule), deepcopy(predefined))
        if not isinstance(batch, list) or any(not isinstance(p, ExperimentParticipant) for p in batch):
            raise ValueError(f"Strategy {name!r} must return a list of ExperimentParticipant objects.")
        if len(batch) != count or any(p.group_id != rule["group_id"] for p in batch):
            raise ValueError(f"Strategy {name!r} must honor the requested count and group.")
        generated.extend(deepcopy(batch))
        validate_participants(predefined + generated, group_ids=group_ids, question_ids=question_ids)
    return generated


def _participants(predefined, generators, label, schema_version="3.0", *,
                  group_ids, question_ids, judgment_encoding="pairwise_preference"):
    """Parse predefined participants, then append the generator's new objects."""
    if schema_version != "3.0" or judgment_encoding != "pairwise_preference":
        raise ValueError("Schema 3.0 requires judgment_encoding=pairwise_preference.")
    if not isinstance(predefined, list):
        raise ValueError("predefined must be a list.")
    group_ids, question_ids = set(group_ids), set(question_ids)
    participants = []
    for item in predefined:
        if not isinstance(item, dict) or not isinstance(item.get("judgments"), dict):
            raise ValueError(f"{label}: each participant requires a judgments object.")
        check_keys(item, {"participant_id", "group_id", "judgments"}, "participant")
        participants.append(ExperimentParticipant(
            participant_id=item["participant_id"], group_id=item["group_id"],
            judgments={qid: parse_preference(value) for qid, value in item["judgments"].items()},
        ))
    validate_participants(participants, group_ids=group_ids, question_ids=question_ids)
    participants.extend(generate_participants(
        generators, participants, group_ids=group_ids, question_ids=question_ids,
    ))
    return participants


def resize_group(participants, operation):
    """Use the same registry to generate the difference to a final group size."""
    check_keys(operation, {"type", "group_id", "count", "strategy",
                           "template_participant_ids", "preserve_existing", "id_pattern"},
               "resize_group")
    count = nonnegative_count(operation["count"], "Final group size")
    gid = operation["group_id"]
    preserve = operation.get("preserve_existing", True)
    if not isinstance(preserve, bool):
        raise ValueError("preserve_existing must be a boolean.")
    original = [p for p in participants if p.group_id == gid]
    if preserve and count < len(original):
        raise ValueError("Shrinking requires preserve_existing=false.")
    kept = deepcopy(original) if preserve else []
    others = deepcopy([p for p in participants if p.group_id != gid])
    additional = count - len(kept)
    # Allocate collision-free IDs first; strategy-specific IDs are temporary.
    used = {p.participant_id for p in others + kept}
    ids = []
    pattern = operation.get("id_pattern", "{group_id}_p{index:02d}")
    for index in range(1, len(used) + additional + 1):
        if len(ids) == additional:
            break
        pid = format_participant_id(pattern, gid, index)
        if pid not in used:
            used.add(pid)
            ids.append(pid)
    if len(ids) != additional:
        raise ValueError("id_pattern cannot produce enough unique participant IDs.")
    temporary_id = "resize"
    while any(p.participant_id.startswith(f"{gid}_{temporary_id}_") for p in original):
        temporary_id += "_"
    rule = {"generator_id": temporary_id, "strategy": operation["strategy"],
            "group_id": gid, "count": additional,
            "template_participant_ids": operation["template_participant_ids"]}
    generated = generate_participants(
        [rule], original, group_ids={gid},
        question_ids=set(original[0].judgments) if original else set(),
    )
    for p, pid in zip(generated, ids):
        p.participant_id = pid
    return others + kept + generated

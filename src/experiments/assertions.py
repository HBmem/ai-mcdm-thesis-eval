"""Validate assertion declarations and evaluate checks against resolved run results."""
import json
import math

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from src.experiments.participants import parse_preference
from src.mcdm.reference import (
    TIE_ATOL, ahp as reference_ahp, topsis as reference_topsis, validate_ratio,
)

# Accepted parameter sets for the checks declared by S01–S10. This registry
# describes contracts, not executable evaluators. Add both when extending checks.
ASSERTION_PARAMETERS = {
    "normalized_weights": (frozenset(),),
    "bounded_topsis_scores": (frozenset(),),
    "submitted_values_in_saaty_domain": (frozenset(),),
    "reciprocal_matrices": (frozenset(),),
    "all_participant_pairwise_matrices_equal": (frozenset(),),
    "all_participant_consistency_ratios": (frozenset(),),
    "session_aggregate_pairwise_matrix_all_ones": (frozenset(),),
    "group_weights_all_equal": (frozenset(),),
    "participant_counts_equal": (frozenset({"counts"}),),
    "group_voting_powers_equal": (frozenset({"expected_powers"}),),
    "group_weights_equal": (frozenset({"reference"}), frozenset({"expected_weights"})),
    "session_weights_equal": (frozenset({"expected_weights"}),),
    "weights_equal": (frozenset({"reference"}),),
    "scores_equal": (frozenset({"reference"}),),
    "winner_sets_equal": (frozenset({"reference"}),),
    "participant_judgments_equal": (frozenset({"reference"}),),
    "alternative_ids_equal_reference_without_winners": (frozenset({"reference", "tie_atol"}),),
    "alternative_ids_equal": (frozenset({"reference"}),),
    "alternative_score_not_greater": (frozenset({"alternative", "reference_alternative"}),),
    "computed_results_match_independent_reference_per_variant": (frozenset(),),
    "criterion_weight_greater_than_reference": (frozenset({"reference", "criterion_id"}),),
    "flagged_participants_retained": (frozenset(),),
    "group_judgments_equal": (frozenset({"expected_judgments"}),),
    "inconsistent_inputs_retained_and_flagged": (frozenset(),),
    "inconsistent_participant_count": (frozenset(),),
    "preference_evidence_equal": (frozenset({"reference"}),),
    "ranking_criterion_ids_equal": (frozenset({"expected_ids"}),),
    "restricted_weights_equal": (frozenset({"reference", "scopes"}),),
    "scoped_scores_equal": (frozenset({"reference", "scopes"}),),
    "scoped_weights_equal": (frozenset({"reference", "scopes"}),),
}

EVALUATORS = {}


@dataclass
class AssertionObservation:
    observed: Any
    passed: bool


def assertion_evaluator(name):
    def register(function):
        if name in EVALUATORS:
            raise ValueError(
                f"Duplicate assertion evaluator: {name}"
            )

        EVALUATORS[name] = function
        return function

    return register


@assertion_evaluator("session_weights_equal")
def session_weights_equal(result, reference, spec):
    observed = _applied_weights(_session(result))
    expected = spec["parameters"]["expected_weights"]

    matches = _close(observed, expected, spec)

    return AssertionObservation(
        observed=_json_evidence(observed),
        passed=bool(matches) == spec["expected"],
    )


@assertion_evaluator("weights_equal")
def weights_equal(result, reference, spec):
    current = _applied_weights(_session(result))
    previous = _applied_weights(_session(reference))

    matches = _close(current, previous, spec)

    return AssertionObservation(
        observed=bool(matches),
        passed=bool(matches) == spec["expected"],
    )


# Entity IDs are local to their scope. Session IDs identify cases, so the
# singleton session uses a common key when comparing different cases.
SCOPES = ("participant", "group", "session")


def _index(items, key, label):
    indexed = {}
    for item in items:
        identifier = key(item)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"Missing or invalid {label} ID: {identifier!r}.")
        if identifier in indexed:
            raise ValueError(f"Duplicate {label} ID: {identifier!r}.")
        indexed[identifier] = item
    return indexed


def _entities(result, scopes=SCOPES, *, require_complete=True):
    if not scopes or len(scopes) != len(set(scopes)) or not set(scopes) <= set(SCOPES):
        raise ValueError("Assertion scopes must be unique participant/group/session scopes.")
    selected = {}
    for scope in scopes:
        if scope == "session":
            if result.mcdm.session is None:
                raise ValueError("Missing session calculation.")
            entities = {"session": result.mcdm.session}
        else:
            values = getattr(result.mcdm, scope + "s")
            entities = _index(values, lambda e: e.entity_id, scope)
            if not entities:
                raise ValueError(f"Missing {scope} calculations.")
        if any(entity.scope != scope for entity in entities.values()):
            raise ValueError(f"Incorrect scope in {scope} calculations.")
        if require_complete:
            session = result.inputs["session"]
            if scope == "session":
                expected = {session["id"]}
            else:
                participants = _index(session["participants"], lambda p: p["id"], "participant")
                expected = (set(participants) if scope == "participant" else
                            {p["stakeholder_group_id"] for p in participants.values()})
            if {entity.entity_id for entity in entities.values()} != expected:
                raise ValueError(f"Incomplete or unexpected {scope} calculations.")
        selected[scope] = entities
    return selected


def _close(current, expected, spec):
    """Compare numeric trees without broadcasting or dropping unmatched IDs."""
    if isinstance(current, dict) or isinstance(expected, dict):
        return (isinstance(current, dict) and isinstance(expected, dict)
                and current.keys() == expected.keys()
                and all(_close(current[k], expected[k], spec) for k in current))
    left, right = np.asarray(current, dtype=float), np.asarray(expected, dtype=float)
    return bool(left.shape == right.shape and np.all(np.isfinite(left))
                and np.all(np.isfinite(right)) and np.allclose(
                    left, right, atol=spec.get("atol", 0.0), rtol=spec.get("rtol", 0.0)))


def _json_evidence(value):
    # Failed numerical checks must still produce portable artifact data.
    if isinstance(value, dict):
        return {k: _json_evidence(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_evidence(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _predicate(matches, spec, observed=None):
    return AssertionObservation(
        observed=_json_evidence(bool(matches) if observed is None else observed),
        passed=bool(matches) == spec["expected"],
    )


def _original_weights(entity):
    if entity.weighting is None:
        raise ValueError(f"Missing weighting for {entity.entity_id!r}.")
    values = _index(entity.weighting.criterion_weights, lambda w: w.criterion_id, "criterion")
    if not values:
        raise ValueError("Missing criterion weights.")
    return {cid: w.weight for cid, w in values.items()}


def _applied_weights(entity):
    values = entity.metadata.applied_criterion_weights
    if not values:
        raise ValueError(f"Missing applied weights for {entity.entity_id!r}.")
    return dict(values)


def _scores(entity):
    if entity.ranking is None:
        raise ValueError(f"Missing ranking for {entity.entity_id!r}.")
    rows = _index(entity.ranking.rankings, lambda r: r.alternative_id, "alternative")
    if not rows:
        raise ValueError("Missing ranking rows.")
    return {aid: row.closeness_coefficient for aid, row in rows.items()}


def _collect(result, extractor, scopes=SCOPES):
    return {scope: {eid: extractor(e) for eid, e in entities.items()}
            for scope, entities in _entities(result, scopes).items()}


def _session(result):
    return _entities(result, ("session",))["session"]["session"]


def _criteria(result):
    return _index(result.inputs["session"]["criteria"], lambda c: c["id"], "criterion")


def _matrix(result, entity):
    ids = list(_criteria(result))
    matrix = np.asarray(entity.pairwise_matrix, dtype=float)
    if matrix.shape != (len(ids), len(ids)) or not ids:
        raise ValueError(f"Incorrect matrix dimensions for {entity.entity_id!r}.")
    # Label both axes to permit criteria reordering across runs.
    return {cid: dict(zip(ids, row.tolist())) for cid, row in zip(ids, matrix)}


def _ratios(result):
    encoding = result.inputs.get("judgment_encoding", "pairwise_preference")
    if encoding not in {"pairwise_preference", "saaty_ratio"}:
        raise ValueError(f"Unsupported judgment encoding: {encoding!r}.")
    participants = _index(result.inputs["session"]["participants"], lambda p: p["id"], "participant")
    questions = _index(result.inputs["session"]["questions"], lambda q: q["id"], "question")
    if not participants or not questions:
        raise ValueError("Missing submitted preference evidence.")
    observed = {}
    for pid, participant in participants.items():
        responses = _index(participant["responses"], lambda r: r["question_id"], "response question")
        if responses.keys() != questions.keys():
            raise ValueError(f"Incomplete responses for {pid!r}.")
        observed[pid] = {
            qid: (float(parse_preference(r["preference"]).ahp_scale)
                  if encoding == "pairwise_preference" else r["preference"])
            for qid, r in responses.items()
        }
    return observed


def _submitted_matrices(result):
    ids = list(_criteria(result))
    positions = {cid: i for i, cid in enumerate(ids)}
    questions = _index(result.inputs["session"]["questions"], lambda q: q["id"], "question")
    pairs = [frozenset((q["left_criterion_id"], q["right_criterion_id"])) for q in questions.values()]
    if (len(pairs) != len(set(pairs)) or set(pairs) !=
            {frozenset(pair) for pair in combinations(ids, 2)}):
        raise ValueError("Questions must cover each criterion pair exactly once.")
    matrices = {}
    for pid, responses in _ratios(result).items():
        matrix = np.ones((len(ids), len(ids)))
        for qid, ratio in responses.items():
            validate_ratio(ratio)
            question = questions[qid]
            i, j = positions[question["left_criterion_id"]], positions[question["right_criterion_id"]]
            matrix[i, j], matrix[j, i] = ratio, 1 / ratio
        matrices[pid] = matrix
    return matrices


@assertion_evaluator("normalized_weights")
def normalized_weights(result, reference, spec):
    observed = _collect(result, lambda e: {"original": _original_weights(e), "applied": _applied_weights(e)})
    matches = all(
        all(math.isfinite(v) and 0 <= v <= 1 for v in weights.values())
        and _close(sum(weights.values()), 1.0, spec)
        for entities in observed.values() for vectors in entities.values() for weights in vectors.values()
    )
    full_ids = set(_criteria(result))
    for entities in _entities(result).values():
        for entity in entities.values():
            selected = entity.metadata.ranking_criterion_ids
            matches &= (set(_original_weights(entity)) == full_ids
                        and bool(selected) and len(selected) == len(set(selected))
                        and set(selected) <= full_ids
                        and set(_applied_weights(entity)) == set(selected))
    return _predicate(matches, spec, observed)


@assertion_evaluator("bounded_topsis_scores")
def bounded_topsis_scores(result, reference, spec):
    observed = _collect(result, _scores)
    expected_ids = _alternative_ids(result)
    matches = all(set(scores) == expected_ids
                  and all(math.isfinite(v) and 0 <= v <= 1 for v in scores.values())
                  for entities in observed.values() for scores in entities.values())
    return _predicate(matches, spec, observed)


@assertion_evaluator("submitted_values_in_saaty_domain")
def submitted_values_in_saaty_domain(result, reference, spec):
    try:
        observed = _ratios(result)
    except ValueError as error:
        if "Invalid preference" not in str(error):
            raise
        return _predicate(False, spec, {"invalid_preference": str(error)})
    valid = True
    for responses in observed.values():
        for ratio in responses.values():
            try:
                validate_ratio(ratio)
            except ValueError:
                valid = False
    return _predicate(valid, spec, observed)


@assertion_evaluator("reciprocal_matrices")
def reciprocal_matrices(result, reference, spec):
    observed = _collect(result, lambda e: _matrix(result, e))
    matches = True
    for entities in _entities(result).values():
        for entity in entities.values():
            a = np.asarray(entity.pairwise_matrix, dtype=float)
            matches &= bool(np.all(np.isfinite(a)) and np.all(a > 0)
                            and _close(np.diag(a), np.ones(len(a)), spec)
                            and _close(a * a.T, np.ones_like(a), spec))
    return _predicate(matches, spec, observed)


@assertion_evaluator("all_participant_pairwise_matrices_equal")
def all_participant_pairwise_matrices_equal(result, reference, spec):
    observed = _collect(result, lambda e: _matrix(result, e), ("participant",))["participant"]
    first = next(iter(observed.values()))
    return _predicate(all(_close(value, first, spec) for value in observed.values()), spec, observed)


@assertion_evaluator("all_participant_consistency_ratios")
def all_participant_consistency_ratios(result, reference, spec):
    observed = _collect(result, _consistency_ratio, ("participant",))["participant"]
    return AssertionObservation(_json_evidence(observed), all(_close(v, spec["expected"], spec) for v in observed.values()))


def _consistency_ratio(entity):
    _original_weights(entity)
    value = entity.weighting.consistency_ratio
    if value is None:
        raise ValueError(f"Missing consistency ratio for {entity.entity_id!r}.")
    return value


@assertion_evaluator("session_aggregate_pairwise_matrix_all_ones")
def session_aggregate_pairwise_matrix_all_ones(result, reference, spec):
    entity = _session(result)
    _matrix(result, entity)
    a = np.asarray(entity.pairwise_matrix)
    return _predicate(_close(a, np.ones_like(a), spec), spec, a)


@assertion_evaluator("group_weights_all_equal")
def group_weights_all_equal(result, reference, spec):
    observed = _collect(result, _original_weights, ("group",))["group"]
    first = next(iter(observed.values()))
    return _predicate(all(_close(v, first, spec) for v in observed.values()), spec, observed)


@assertion_evaluator("participant_counts_equal")
def participant_counts_equal(result, reference, spec):
    session = result.inputs["session"]
    counts = {gid: 0 for gid in _index(session["stakeholder_groups"], lambda g: g["id"], "group")}
    for participant in _index(session["participants"], lambda p: p["id"], "participant").values():
        counts[participant["stakeholder_group_id"]] += 1
    return _predicate(counts == spec["parameters"]["counts"], spec, counts)


@assertion_evaluator("group_voting_powers_equal")
def group_voting_powers_equal(result, reference, spec):
    groups = _index(result.inputs["session"]["stakeholder_groups"], lambda g: g["id"], "group")
    observed = {gid: g["voting_power"] for gid, g in groups.items()}
    return _predicate(_close(observed, spec["parameters"]["expected_powers"], spec), spec, observed)


@assertion_evaluator("group_weights_equal")
def group_weights_equal(result, reference, spec):
    observed = _collect(result, _original_weights, ("group",))["group"]
    if "reference" in spec["parameters"]:
        expected = _collect(reference, _original_weights, ("group",))["group"]
    else:
        expected = {gid: spec["parameters"]["expected_weights"] for gid in observed}
    return _predicate(_close(observed, expected, spec), spec, observed)


@assertion_evaluator("scores_equal")
def scores_equal(result, reference, spec):
    observed = _scores(_session(result))
    return _predicate(_close(observed, _scores(_session(reference)), spec), spec, observed)


@assertion_evaluator("scoped_weights_equal")
def scoped_weights_equal(result, reference, spec):
    scopes = spec["parameters"]["scopes"]
    observed = _collect(result, _applied_weights, scopes)
    return _predicate(_close(observed, _collect(reference, _applied_weights, scopes), spec), spec, observed)


@assertion_evaluator("scoped_scores_equal")
def scoped_scores_equal(result, reference, spec):
    scopes = spec["parameters"]["scopes"]
    observed = _collect(result, _scores, scopes)
    return _predicate(_close(observed, _collect(reference, _scores, scopes), spec), spec, observed)


@assertion_evaluator("restricted_weights_equal")
def restricted_weights_equal(result, reference, spec):
    scopes = spec["parameters"]["scopes"]
    observed = _collect(result, _applied_weights, scopes)
    previous = _collect(reference, _applied_weights, scopes)
    expected = {}
    for scope, entities in observed.items():
        if entities.keys() != previous[scope].keys():
            return _predicate(False, spec, observed)
        expected[scope] = {}
        for eid, weights in entities.items():
            source = previous[scope][eid]
            if not weights.keys() <= source.keys():
                return _predicate(False, spec, observed)
            total = sum(source[cid] for cid in weights)
            if not math.isfinite(total) or total <= 0:
                raise ValueError("Reference weights have no positive retained mass.")
            expected[scope][eid] = {cid: source[cid] / total for cid in weights}
    return _predicate(_close(observed, expected, spec), spec, observed)


@assertion_evaluator("ranking_criterion_ids_equal")
def ranking_criterion_ids_equal(result, reference, spec):
    expected = spec["parameters"]["expected_ids"]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Expected ranking criterion IDs must be nonempty and unique.")
    observed = _collect(result, lambda e: e.metadata.ranking_criterion_ids)
    return _predicate(all(len(ids) == len(set(ids)) and set(ids) == set(expected)
                          for entities in observed.values() for ids in entities.values()), spec, observed)


def _winners(result, tie_atol=TIE_ATOL):
    scores = _scores(_session(result))
    if not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("Winner selection requires finite scores.")
    best = max(scores.values())
    return sorted(aid for aid, score in scores.items() if abs(score - best) <= tie_atol)


@assertion_evaluator("winner_sets_equal")
def winner_sets_equal(result, reference, spec):
    observed = _winners(result)
    return _predicate(observed == _winners(reference), spec, observed)


def _alternative_ids(result):
    return set(_index(result.inputs["session"]["alternatives"], lambda a: a["id"], "alternative"))


@assertion_evaluator("alternative_ids_equal")
def alternative_ids_equal(result, reference, spec):
    observed = _alternative_ids(result)
    return _predicate(observed == _alternative_ids(reference), spec, sorted(observed))


@assertion_evaluator("alternative_ids_equal_reference_without_winners")
def alternative_ids_equal_reference_without_winners(result, reference, spec):
    observed = _alternative_ids(result)
    expected = _alternative_ids(reference) - set(_winners(reference, spec["parameters"]["tie_atol"]))
    return _predicate(observed == expected, spec, sorted(observed))


@assertion_evaluator("alternative_score_not_greater")
def alternative_score_not_greater(result, reference, spec):
    scores = _scores(_session(result))
    left = scores[spec["parameters"]["alternative"]]
    right = scores[spec["parameters"]["reference_alternative"]]
    return _predicate(math.isfinite(left) and math.isfinite(right)
                      and (left <= right or _close(left, right, spec)), spec,
                      {"alternative_score": left, "reference_score": right})


@assertion_evaluator("criterion_weight_greater_than_reference")
def criterion_weight_greater_than_reference(result, reference, spec):
    cid = spec["parameters"]["criterion_id"]
    left, right = _applied_weights(_session(result))[cid], _applied_weights(_session(reference))[cid]
    return _predicate(math.isfinite(left) and math.isfinite(right)
                      and left > right and not _close(left, right, spec), spec,
                      {"current": left, "reference": right})


def _judgments(result):
    ratios = _ratios(result)
    return {p["id"]: {"group_id": p["stakeholder_group_id"], "responses": ratios[p["id"]]}
            for p in result.inputs["session"]["participants"]}


@assertion_evaluator("participant_judgments_equal")
def participant_judgments_equal(result, reference, spec):
    observed = _judgments(result)
    return _predicate(observed == _judgments(reference), spec, observed)


@assertion_evaluator("group_judgments_equal")
def group_judgments_equal(result, reference, spec):
    judgments = _judgments(result)
    observed, matches = {}, True
    for gid, declarations in spec["parameters"]["expected_judgments"].items():
        expected = {qid: float(parse_preference(code).ahp_scale) for qid, code in declarations.items()}
        members = {pid: p["responses"] for pid, p in judgments.items() if p["group_id"] == gid}
        if not members:
            raise ValueError(f"No submitted participants for group {gid!r}.")
        observed[gid] = members
        matches &= all(_close(responses, expected, spec) for responses in members.values())
    return _predicate(matches, spec, observed)


def _preference_evidence(result):
    criteria = _criteria(result)
    questions = _index(result.inputs["session"]["questions"], lambda q: q["id"], "question")
    groups = _index(result.inputs["session"]["stakeholder_groups"], lambda g: g["id"], "group")
    return {"criteria": criteria, "questions": questions, "groups": groups, "judgments": _judgments(result)}


@assertion_evaluator("preference_evidence_equal")
def preference_evidence_equal(result, reference, spec):
    observed = _preference_evidence(result)
    previous = _preference_evidence(reference)
    inputs_equal = observed == previous

    def calculations(run_result):
        return _collect(run_result, lambda entity: {
            "matrix": _matrix(run_result, entity),
            "weights": _original_weights(entity),
            "consistency_ratio": _consistency_ratio(entity),
            "consistency_threshold": entity.weighting.consistency_threshold,
        })

    observed["calculations"] = calculations(result)
    previous["calculations"] = calculations(reference)
    return _predicate(inputs_equal and _close(observed["calculations"], previous["calculations"], spec), spec, observed)


def _reference_ahp(result, matrix):
    processing = result.inputs["processing"]
    config = processing["weighting_method"]
    if config["method"] != "ahp" or set(config["parameters"]) - {"derivation"}:
        raise NotImplementedError("Independent reference supports AHP with a derivation parameter only.")
    return reference_ahp(matrix, config["parameters"].get("derivation", "geometric"),
                         processing["consistency_method"]["threshold"])


def _inconsistency_evidence(result, spec):
    matrices = _submitted_matrices(result)
    entities = _entities(result, ("participant",), require_complete=False)["participant"]
    threshold = result.inputs["processing"]["consistency_method"]["threshold"]
    evidence = {}
    judgments = _judgments(result)
    retained = matrices.keys() == entities.keys()
    for pid, matrix in matrices.items():
        expected_ratio = _reference_ahp(result, matrix)["consistency_ratio"]
        flagged = expected_ratio > threshold
        entity = entities.get(pid)
        consistent = False
        if entity is not None:
            ratio = _consistency_ratio(entity)
            consistent = (_close(entity.pairwise_matrix, matrix, spec)
                          and math.isfinite(ratio) and ratio >= 0
                          and entity.metadata.group_id == judgments[pid]["group_id"]
                          and entity.weighting.consistency_threshold == threshold
                          and entity.weighting.exceeds_consistency_threshold == flagged)
        retained &= consistent
        evidence[pid] = {"expected_ratio": expected_ratio, "flagged": flagged, "retained": bool(consistent)}
    return evidence, bool(retained)


@assertion_evaluator("inconsistent_participant_count")
def inconsistent_participant_count(result, reference, spec):
    ratios = _collect(result, _consistency_ratio, ("participant",))["participant"]
    threshold = result.inputs["processing"]["consistency_method"]["threshold"]
    if not all(math.isfinite(v) for v in ratios.values()):
        raise ValueError("Inconsistency counting requires finite consistency ratios.")
    count = sum(v > threshold for v in ratios.values())
    return AssertionObservation(count, type(spec["expected"]) is int and count == spec["expected"])


@assertion_evaluator("flagged_participants_retained")
@assertion_evaluator("inconsistent_inputs_retained_and_flagged")
def flagged_participants_retained(result, reference, spec):
    observed, matches = _inconsistency_evidence(result, spec)
    return _predicate(matches, spec, observed)


@assertion_evaluator("computed_results_match_independent_reference_per_variant")
def computed_results_match_independent_reference_per_variant(result, reference, spec):
    """Recompute evidence without calling the production MCDM service."""
    processing = result.inputs["processing"]
    aggregation = processing["aggregation_method"]
    supported = {"within_group": "geometric_mean", "between_groups": "weighted_geometric_mean",
                 "group_power_normalization": "represented_groups", "empty_group_policy": "excluded"}
    if aggregation != supported:
        raise NotImplementedError("Independent reference requires geometric aggregation with represented groups and excluded empty groups.")
    ranking = processing["ranking_method"]
    if (ranking["method"] != "topsis" or set(ranking["parameters"]) - {"normalization"}
            or ranking["parameters"].get("normalization", "vector") != "vector"):
        raise NotImplementedError("Independent reference supports vector-normalized TOPSIS only.")
    session = result.inputs["session"]
    criteria = _criteria(result)
    participants = _index(session["participants"], lambda p: p["id"], "participant")
    groups = _index(session["stakeholder_groups"], lambda g: g["id"], "group")
    if any(p["stakeholder_group_id"] not in groups for p in participants.values()):
        raise ValueError("Participant references an unknown group.")
    matrices = {"participant": _submitted_matrices(result), "group": {}}
    powers = []
    for gid, group in groups.items():
        members = [matrices["participant"][pid] for pid, p in participants.items() if p["stakeholder_group_id"] == gid]
        if members:
            matrices["group"][gid] = np.exp(np.mean(np.log(members), axis=0))
            powers.append(group["voting_power"])
    powers = np.asarray(powers, dtype=float)
    if not len(powers) or not np.all(np.isfinite(powers)) or np.any(powers < 0) or powers.sum() <= 0:
        raise ValueError("Independent aggregation requires positive represented voting power.")
    matrices["session"] = {"session": np.exp(np.tensordot(powers / powers.sum(),
                                         np.log(list(matrices["group"].values())), axes=1))}
    # Resolve ranking interventions from the saved declarations, rather than
    # trusting the weights or criterion IDs being checked.
    selected = list(criteria)
    overrides = {}
    for operation in result.inputs.get("case_metadata", {}).get("runtime_operations", []):
        if operation["type"] == "remove_criteria":
            selected = [cid for cid in selected if cid not in operation["criterion_ids"]]
        elif operation["type"] == "set_criterion_weights":
            for scope in operation["scopes"]:
                overrides[scope] = operation["weights"]
    alternatives = _index(session["alternatives"], lambda a: a["id"], "alternative")
    alternative_values = {
        aid: {cid: item["value"] for cid, item in _index(a["criterion_values"], lambda v: v["criterion_id"], "alternative criterion").items()}
        for aid, a in alternatives.items()
    }
    entities = _entities(result, require_complete=False)
    observed, matches = {}, True
    for scope in SCOPES:
        observed[scope] = {}
        if entities[scope].keys() != matrices[scope].keys():
            matches = False
        for eid, matrix in matrices[scope].items():
            if eid not in entities[scope]:
                observed[scope][eid] = {"missing": True}
                continue
            entity = entities[scope][eid]
            weights = _reference_ahp(result, matrix)
            original = dict(zip(criteria, weights["weights"]))
            applied = {cid: original[cid] for cid in selected} if scope not in overrides else dict(overrides[scope])
            total = sum(applied.values())
            if not applied or total <= 0 or applied.keys() != set(selected):
                raise ValueError("Independent ranking weights must cover selected criteria with positive mass.")
            applied = {cid: value / total for cid, value in applied.items()}
            expected_ranking = reference_topsis(
                [{"id": aid, "values": values} for aid, values in alternative_values.items()],
                [criteria[cid] for cid in selected], [applied[cid] for cid in selected])
            scores = {r["alternative_id"]: r["score"] for r in expected_ranking["rankings"]}
            # Exact ties receive competition ranks in the current result model.
            ordered = expected_ranking["rankings"]
            ranks = {r["alternative_id"]: 1 + sum(other["score"] > r["score"] for other in ordered) for r in ordered}
            actual_rows = _index(entity.ranking.rankings, lambda r: r.alternative_id, "alternative") if entity.ranking else {}
            checks = {
                "matrix": _close(entity.pairwise_matrix, matrix, spec),
                "weights": _close(_original_weights(entity), original, spec),
                "consistency_ratio": _close(_consistency_ratio(entity), weights["consistency_ratio"], spec),
                "consistency_threshold": entity.weighting.consistency_threshold == weights["threshold"],
                "applied_weights": _close(_applied_weights(entity), applied, spec),
                "ranking_criteria": (len(entity.metadata.ranking_criterion_ids) == len(selected)
                                     and set(entity.metadata.ranking_criterion_ids) == set(selected)),
                "scores": _close(_scores(entity), scores, spec),
                "ranks": {aid: r.rank for aid, r in actual_rows.items()} == ranks,
            }
            observed[scope][eid] = checks
            matches &= all(checks.values())
    return _predicate(matches, spec, observed)


def require_numerical_result(result):
    if result.execution.get("numerical_status") != "completed":
        raise ValueError(
            "Referenced run has no completed numerical results."
        )

    result.validate_numerical_complete()
    return result


def evaluate_declared_assertions(
    run,
    result,
    completed,
    *,
    stage,
):
    records = []

    for spec in run.experiment.expected_assertions:
        reference_case = spec.get(
            "parameters", {}
        ).get("reference")

        declared_stage = spec.get(
            "stage",
            "comparison" if reference_case else "calculation",
        )

        if declared_stage != stage:
            continue

        record = {
            **spec,
            "observed": None,
            "status": "error",
            "reason": None,
        }

        if spec.get("deferred", False):
            record.update(status="deferred", reason=spec.get("reason"))
            records.append(record)
            continue

        try:
            handler = EVALUATORS.get(spec["check"])

            if handler is None:
                raise LookupError(
                    f"No evaluator for {spec['check']!r}."
                )

            reference = None

            if reference_case is not None:
                reference_key = run.reference_runs[reference_case]
                reference = require_numerical_result(
                    completed[reference_key]
                )
                record["reference_run_id"] = reference.run_id

            observation = handler(result, reference, spec)

            record["observed"] = observation.observed
            record["status"] = (
                "passed" if observation.passed else "failed"
            )

        except Exception as error:
            record["reason"] = (
                f"{type(error).__name__}: {error}"
            )

        records.append(record)

    return records

def validate_assertions(specs, *, criterion_ids, group_ids):
    """Validate named check declarations and return their referenced case keys."""
    # Require a list of assertion objects before inspecting individual fields.
    if not isinstance(specs, list):
        raise ValueError("Assertions must be a list.")
    ids, references = set(), []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Each assertion must be an object.")
        # Unique IDs make each assertion result unambiguous within its case.
        identifier = spec.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("Assertion IDs must be unique nonempty strings within a case.")
        ids.add(identifier)
        # Registered names and exact parameter sets catch misspellings, missing
        # arguments, and unsupported arguments before execution.
        name = spec.get("check")
        if not isinstance(name, str) or name not in ASSERTION_PARAMETERS:
            raise ValueError(f"Unknown assertion check: {name!r}.")
        parameters = spec.get("parameters", {})
        if not isinstance(parameters, dict) or frozenset(parameters) not in ASSERTION_PARAMETERS[name]:
            raise ValueError(f"Invalid parameters for assertion {name!r}.")
        # Winner membership uses an absolute score tolerance. Reject booleans
        # explicitly because Python treats bool as a numeric subtype.
        if "tie_atol" in parameters:
            value = parameters["tie_atol"]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError("Assertion tie_atol must be finite and nonnegative.")
        # Expectations must be serializable without NaN or infinity so they can
        # be stored in portable JSON artifacts.
        if "expected" not in spec:
            raise ValueError(f"Assertion {identifier!r} requires expected.")
        try:
            json.dumps(spec["expected"], allow_nan=False)
        except (ValueError, TypeError) as error:
            raise ValueError("Assertion expectations must be finite JSON data.") from error
        # Comparison tolerances must be finite, nonnegative numbers; zero is valid.
        for field in ("atol", "rtol"):
            value = spec.get(field, 0)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"Assertion {field} must be finite and nonnegative.")
        # Severity controls whether a failed check is an error or a warning.
        if spec.get("severity", "error") not in {"error", "warning"}:
            raise ValueError("Assertion severity must be error or warning.")
        # Checks using another case belong to the comparison stage. Collect
        # their nonempty references for the planner to resolve against its catalog.
        reference = parameters.get("reference")
        expected_stage = "comparison" if reference is not None else "calculation"
        if spec.get("stage", expected_stage) != expected_stage:
            raise ValueError(f"Assertion {identifier!r} must use stage={expected_stage}.")
        if reference is not None:
            if not isinstance(reference, str) or not reference:
                raise ValueError("Assertion reference must be a nonempty case key.")
            references.append(reference)
        elif "reference" in parameters:
            raise ValueError("Assertion reference cannot be null.")
        # Expected mappings must cover exactly the known groups or criteria so
        # missing entities cannot silently escape a check.
        for field, keys in (("counts", group_ids), ("expected_powers", group_ids),
                            ("expected_weights", criterion_ids)):
            if field not in parameters:
                continue
            values = parameters[field]
            if not isinstance(values, dict) or set(values) != set(keys):
                raise ValueError(f"Assertion {field} must cover exactly the expected entity IDs.")
            for value in values.values():
                # Counts are whole numbers; powers and weights may be fractional.
                if field == "counts":
                    valid = not isinstance(value, bool) and isinstance(value, int) and value >= 0
                else:
                    valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                             and math.isfinite(value) and value >= 0)
                if not valid:
                    raise ValueError(f"Invalid assertion value in {field}.")
    # Deduplicate references while preserving their declaration order.
    return tuple(dict.fromkeys(references))

"""PyDecision adapters and numerical evaluation of one resolved experiment.

Registries are the extension points. Calculation is separate from batch storage,
assertions, comparisons, and AI. No result here claims an experiment has passed.
"""
from dataclasses import dataclass
from importlib.metadata import version
from typing import Callable

import numpy as np
from pyDecision.algorithm import ahp_method, topsis_method

from src.mcdm.enums import (
    WeightingMethod, RankingMethod, MeanMethod, GroupPowerMethod,
    EmptyGroupPolicy, OnExceedancePolicy, PairwisePreference,
)


DERIVATIONS = {
    "mean": "mean", "m": "mean", "geometric": "geometric", "g": "geometric",
    "max_eigen": "max_eigen", "me": "max_eigen",
}
TIE_ATOL = 1e-12


@dataclass(frozen=True)
class MethodAdapter:
    validate: Callable
    calculate: Callable


def _ahp_parameters(parameters):
    if (not isinstance(parameters, dict) or set(parameters) - {"derivation"}
            or parameters.get("derivation", "geometric") not in DERIVATIONS):
        raise ValueError("Unsupported AHP parameters or derivation.")


def _topsis_parameters(parameters):
    if (not isinstance(parameters, dict) or set(parameters) - {"normalization"}
            or parameters.get("normalization", "vector") != "vector"):
        raise ValueError("Only vector TOPSIS normalization is supported.")


def _matrix(matrix):
    a = np.asarray(matrix, dtype=float)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not 2 <= len(a) <= 15
            or not np.all(np.isfinite(a)) or np.any(a <= 0)
            or not np.allclose(np.diag(a), 1, atol=1e-12, rtol=0)
            or not np.allclose(a * a.T, 1, atol=1e-10, rtol=0)):
        raise ValueError("AHP requires a finite positive reciprocal 2–15 criterion matrix.")
    return a


def labeled(values, rows, columns):
    return {"row_ids": list(rows), "column_ids": list(columns),
            "values": np.asarray(values, dtype=float).tolist()}


def calculate_ahp(matrix, parameters, threshold):
    """Use PyDecision at participant, group, and session level alike."""
    _ahp_parameters(parameters)
    matrix = _matrix(matrix)
    derivation = DERIVATIONS[parameters.get("derivation", "geometric")]
    # PyDecision computes two-criterion weights, but its RI[2]=0 makes CR undefined.
    # Keep its weights and explicitly apply the documented CR=0 convention.
    with np.errstate(divide="ignore", invalid="ignore"):
        weights, cr = ahp_method(matrix.copy(), wd=derivation)
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (len(matrix),) or not np.all(np.isfinite(weights))
            or np.any(weights <= 0) or not np.isclose(weights.sum(), 1, atol=1e-10, rtol=0)):
        raise ValueError("PyDecision returned invalid AHP weights.")
    cr = 0.0 if len(matrix) == 2 else float(cr)
    if not np.isfinite(cr) or cr < -1e-10:
        raise ValueError("PyDecision returned an invalid consistency ratio.")
    cr = max(0.0, cr)  # Remove negative floating-point noise, not inconsistency.
    eigenvalue = (float(np.max(np.linalg.eigvals(matrix).real)) if derivation == "max_eigen"
                  else float(np.mean((matrix @ weights) / weights)))
    return {
        "weights": weights.tolist(), "consistency_ratio": cr,
        "flagged": bool(cr > threshold), "threshold": float(threshold),
        "wd": derivation, "lambda_max": eigenvalue,
        "consistency_convention": ("principal_eigenvalue" if derivation == "max_eigen"
                                    else "mean_component_ratio"),
        "two_criterion_cr_convention": "zero" if len(matrix) == 2 else None,
    }


def calculate_topsis(matrix, weights, directions, parameters):
    """Get scores from PyDecision; reconstruct labeled intermediates for audit.

    This first adapter rejects undefined zero-column/identical-option cases.
    It never replaces an invalid library output with a successful-looking score.
    """
    _topsis_parameters(parameters)
    x, w = np.asarray(matrix, dtype=float), np.asarray(weights, dtype=float)
    if (x.ndim != 2 or not x.shape[0] or not x.shape[1]
            or not np.all(np.isfinite(x)) or w.shape != (x.shape[1],)
            or not np.all(np.isfinite(w)) or np.any(w < 0)
            or not np.isclose(w.sum(), 1, atol=1e-10, rtol=0)
            or len(directions) != x.shape[1] or any(d not in {"max", "min"} for d in directions)):
        raise ValueError("Invalid TOPSIS inputs.")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        norms = np.linalg.norm(x, axis=0)
        if np.any(norms == 0):
            raise ValueError("PyDecision TOPSIS cannot normalize an all-zero criterion column.")
        normalized = x / norms
        weighted = normalized * w
        benefit = np.array([d == "max" for d in directions])
        best = np.where(benefit, weighted.max(axis=0), weighted.min(axis=0))
        worst = np.where(benefit, weighted.min(axis=0), weighted.max(axis=0))
        db = np.linalg.norm(weighted - best, axis=1)
        dw = np.linalg.norm(weighted - worst, axis=1)
        if np.any(db + dw == 0):
            raise ValueError("PyDecision TOPSIS is undefined when all effective alternatives coincide.")
        scores = np.asarray(topsis_method(
            x.copy(), w.copy(), list(directions), graph=False, verbose=False,
        ), dtype=float)
    if (scores.shape != (x.shape[0],) or not np.all(np.isfinite(scores))
            or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("PyDecision returned invalid TOPSIS scores.")
    return {"scores": scores, "normalized": normalized, "weighted": weighted,
            "ideal_best": best, "ideal_worst": worst,
            "distance_best": db, "distance_worst": dw}


def geometric_mean(matrices):
    """Aggregate judgments before generating group weights (AIJ)."""
    return np.exp(np.log(np.asarray(matrices, dtype=float)).mean(axis=0))


def weighted_geometric_mean(matrices, powers):
    powers = np.asarray(powers, dtype=float)
    if (powers.shape != (len(matrices),) or not np.all(np.isfinite(powers))
            or np.any(powers < 0) or powers.sum() <= 0):
        raise ValueError("Represented group powers need a finite positive total.")
    return np.exp(np.tensordot(powers / powers.sum(), np.log(matrices), axes=1))


# Add an enum member and a validated adapter here when implementing a new method.
# Aggregation registries have separate signatures for each level on purpose.
WEIGHTING_METHODS = {WeightingMethod.AHP: MethodAdapter(_ahp_parameters, calculate_ahp)}
RANKING_METHODS = {RankingMethod.TOPSIS: MethodAdapter(_topsis_parameters, calculate_topsis)}
WITHIN_GROUP_METHODS = {MeanMethod.GEOMETRIC_MEAN: geometric_mean}
BETWEEN_GROUP_METHODS = {MeanMethod.WEIGHTED_GEOMETRIC_MEAN: weighted_geometric_mean}


def validate_methods(process):
    """Shared capability check for preparation and execution; no calculations."""
    weighting, ranking, aggregation = (
        process.weighting_method, process.ranking_method, process.aggregation_method,
    )
    if weighting.method not in WEIGHTING_METHODS or ranking.method not in RANKING_METHODS:
        raise ValueError("Unsupported weighting or ranking method.")
    WEIGHTING_METHODS[weighting.method].validate(weighting.parameters)
    RANKING_METHODS[ranking.method].validate(ranking.parameters)
    if (aggregation.within_group not in WITHIN_GROUP_METHODS
            or aggregation.between_groups not in BETWEEN_GROUP_METHODS
            or aggregation.group_power_normalization != GroupPowerMethod.REPRESENTED_GROUPS
            or aggregation.empty_group_policy != EmptyGroupPolicy.EXCLUDED):
        raise ValueError("Unsupported aggregation configuration.")
    consistency = process.consistency_method
    if (isinstance(consistency.threshold, bool)
            or not isinstance(consistency.threshold, (int, float))
            or not np.isfinite(consistency.threshold) or consistency.threshold < 0
            or consistency.on_exceedance != OnExceedancePolicy.RETAIN_AND_FLAG):
        raise ValueError("Unsupported consistency configuration.")


def participant_matrix(session, participant):
    cids = [c.id for c in session.criteria]
    positions = {cid: i for i, cid in enumerate(cids)}
    questions = {q.id: q for q in session.questions}
    response_ids = [r.question_id for r in participant.responses]
    if len(response_ids) != len(questions) or set(response_ids) != set(questions):
        raise ValueError(f"{participant.id}: incomplete or duplicated judgments.")
    matrix = np.ones((len(cids), len(cids)))
    for response in participant.responses:
        if not isinstance(response.preference, PairwisePreference):
            raise ValueError("Judgments must be parsed PairwisePreference members.")
        q = questions[response.question_id]
        i, j = positions[q.left_criterion_id], positions[q.right_criterion_id]
        ratio = response.preference.ahp_scale
        matrix[i, j], matrix[j, i] = ratio, 1 / ratio
    return _matrix(matrix)


def evaluate_experiment(experiment):
    """Calculate weights and rankings for all represented entities in one case."""
    process, session = experiment.processing, experiment.session
    validate_methods(process)
    cids, aids = [c.id for c in session.criteria], [a.id for a in session.alternatives]
    if not session.participants or not aids:
        raise ValueError("Participants and alternatives are required.")
    for label, ids in (("criteria", cids), ("alternatives", aids),
                       ("participants", [p.id for p in session.participants]),
                       ("groups", [g.id for g in session.stakeholder_groups])):
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {label} IDs.")
    pconfig, rconfig = process.weighting_method, process.ranking_method
    weight_handler = WEIGHTING_METHODS[pconfig.method].calculate
    rank_handler = RANKING_METHODS[rconfig.method].calculate

    def weigh(identifier, matrix, **metadata):
        _matrix(matrix)
        return {"id": identifier, **metadata, "matrix": labeled(matrix, cids, cids),
                **weight_handler(matrix, pconfig.parameters, process.consistency_method.threshold)}

    individuals = [weigh(p.id, participant_matrix(session, p), group_id=p.stakeholder_group_id)
                   for p in session.participants]
    group_ids = {g.id for g in session.stakeholder_groups}
    if any(p["group_id"] not in group_ids for p in individuals):
        raise ValueError("A participant references an unknown group.")
    groups, excluded = [], []
    within = WITHIN_GROUP_METHODS[process.aggregation_method.within_group]
    between = BETWEEN_GROUP_METHODS[process.aggregation_method.between_groups]
    for group in session.stakeholder_groups:
        members = [p for p in individuals if p["group_id"] == group.id]
        if not members:
            excluded.append(group.id)
            continue
        matrix = within([p["matrix"]["values"] for p in members])
        groups.append(weigh(group.id, matrix, participant_count=len(members),
                            voting_power=group.voting_power))
    if not groups:
        raise ValueError("No represented stakeholder groups.")
    powers = np.asarray([g["voting_power"] for g in groups], dtype=float)
    # The registered between-group handler validates and normalizes group powers.
    matrix = between([g["matrix"]["values"] for g in groups], powers)
    for group, power in zip(groups, powers / powers.sum()):
        group["effective_voting_power"] = float(power)
    aggregate = weigh(session.id, matrix)

    rows = []
    for alternative in session.alternatives:
        values = {v.criterion_id: v.value for v in alternative.criterion_values}
        if len(alternative.criterion_values) != len(cids) or set(values) != set(cids):
            raise ValueError("Every alternative must cover each criterion exactly once.")
        rows.append([values[cid] for cid in cids])
    decision = np.asarray(rows, dtype=float)
    directions = [c.direction.value for c in session.criteria]

    def rank(preference):
        numerical = rank_handler(decision, preference["weights"], directions, rconfig.parameters)
        scores = numerical["scores"]
        order = sorted(range(len(aids)), key=lambda i: (-float(scores[i]), i))
        statistical = np.empty(len(aids))
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and abs(scores[order[start]] - scores[order[end]]) <= TIE_ATOL:
                end += 1
            statistical[order[start:end]] = (start + 1 + end) / 2
            start = end
        return {
            "criterion_ids": cids.copy(),
            "applied_weights": dict(zip(cids, preference["weights"])),
            "decision_matrix": labeled(decision, aids, cids),
            "normalized_matrix": labeled(numerical["normalized"], aids, cids),
            "weighted_matrix": labeled(numerical["weighted"], aids, cids),
            "ideal_best": dict(zip(cids, numerical["ideal_best"].tolist())),
            "ideal_worst": dict(zip(cids, numerical["ideal_worst"].tolist())),
            "rankings": [{"alternative_id": aids[i], "rank": rank,
                          "statistical_rank": float(statistical[i]), "score": float(scores[i]),
                          "distance_best": float(numerical["distance_best"][i]),
                          "distance_worst": float(numerical["distance_worst"][i])}
                         for rank, i in enumerate(order, 1)],
            "winner_ids": [aids[i] for i in order if abs(scores[i] - scores[order[0]]) <= TIE_ATOL],
            "tie_tolerance": TIE_ATOL,
        }

    return {
        "preferences": {"criterion_ids": cids, "participants": individuals, "groups": groups,
                        "session": aggregate, "excluded_group_ids": excluded, "source": "pyDecision"},
        "applied_weights": dict(zip(cids, aggregate["weights"])),
        "participant_rankings": {p["id"]: rank(p) for p in individuals},
        "group_rankings": {g["id"]: rank(g) for g in groups},
        "topsis": rank(aggregate),
        "engine": {"name": "pyDecision", "version": version("pyDecision"),
                   "weighting_method": pconfig.method.value,
                   "ranking_method": rconfig.method.value,
                   "within_group": process.aggregation_method.within_group.value,
                   "between_groups": process.aggregation_method.between_groups.value,
                   "processing": {"weighting": dict(pconfig.parameters), "ranking": dict(rconfig.parameters)},
                   "aggregation_implementation": "local judgment-matrix adapters",
                   "intermediate_matrices": "locally reconstructed from PyDecision inputs",
                   "undefined_topsis_policy": "reject"},
    }

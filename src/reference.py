"""Independent numerical reference. No application calculation or pyDecision imports.

RI values follow the table documented in the installed AHP implementation; mean
and geometric CR use the mean component-ratio eigenvalue estimate. This explicit
convention matters for inconsistent matrices, where derivations can disagree.
"""
from itertools import combinations
from numbers import Real

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

RI = [0, 0, 0, .58, .90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.59]
SAATY = tuple([1 / i for i in range(9, 1, -1)] + list(range(1, 10)))
DERIVATIONS = {"mean": "mean", "m": "mean", "geometric": "geometric", "g": "geometric", "max_eigen": "max_eigen", "me": "max_eigen"}
TIE_ATOL = 1e-12


def validate_ratio(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not any(np.isclose(value, x, rtol=0, atol=1e-12) for x in SAATY):
        raise ValueError(f"Invalid Saaty ratio: {value!r}")


def quantize_ratio(value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Target ratios must be finite and positive.")
    distances = [abs(np.log(value / x)) for x in SAATY]
    best = min(distances)
    return float(min((x for x, d in zip(SAATY, distances) if abs(d - best) <= 1e-14), key=lambda x: abs(np.log(x))))


def labeled(values, rows, columns):
    return {"row_ids": list(rows), "column_ids": list(columns), "values": np.asarray(values).tolist()}


def validate_matrix(matrix):
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not 2 <= len(a) <= 15:
        raise ValueError("AHP requires a square matrix with 2–15 criteria.")
    if not np.all(np.isfinite(a)) or np.any(a <= 0) or not np.allclose(np.diag(a), 1) or not np.allclose(a * a.T, 1):
        raise ValueError("AHP matrix must be finite, positive, reciprocal, with unit diagonal.")
    return a


def ahp(matrix, wd="mean", threshold=.10):
    a = validate_matrix(matrix)
    if wd not in DERIVATIONS:
        raise ValueError(f"Unsupported AHP derivation: {wd!r}")
    method = DERIVATIONS[wd]
    if method == "mean":
        w = (a / a.sum(axis=0)).mean(axis=1)
    elif method == "geometric":
        w = np.exp(np.log(a).mean(axis=1))
    else:
        values, vectors = np.linalg.eig(a)
        index = np.argmax(values.real)
        w = vectors[:, index].real
    w = w / w.sum()
    eigenvalue = float(values[index].real) if method == "max_eigen" else float(np.mean((a @ w) / w))
    ci = max(0., (eigenvalue - len(a)) / (len(a) - 1))
    cr = ci / RI[len(a)] if len(a) > 2 else 0.
    return {"weights": w.tolist(), "lambda_max": eigenvalue, "consistency_index": ci,
            "consistency_ratio": cr, "flagged": bool(cr > threshold), "wd": wd,
            "threshold": threshold, "ri": RI[len(a)],
            "consistency_convention": "principal_eigenvalue" if method == "max_eigen" else "mean_component_ratio"}


def preferences(inputs):
    criteria = [c["id"] for c in inputs["preference_criteria"]]
    positions = {key: i for i, key in enumerate(criteria)}
    questions = {q["id"]: q for q in inputs["questions"]}
    wd = inputs["processing"]["wd"]
    threshold = inputs["consistency_threshold"]
    individual = []
    for p in inputs["participants"]:
        if set(p["judgments"]) != set(questions):
            raise ValueError(f"Incomplete judgments for {p['id']}")
        a = np.ones((len(criteria), len(criteria)))
        for key, ratio in p["judgments"].items():
            validate_ratio(ratio)
            q = questions[key]
            i, j = positions[q["left_criterion_id"]], positions[q["right_criterion_id"]]
            a[i, j], a[j, i] = ratio, 1 / ratio
        individual.append({"id": p["id"], "group_id": p["group_id"], "matrix": labeled(a, criteria, criteria), **ahp(a, wd, threshold)})
    grouped = []
    for g in inputs["groups"]:
        members = [p for p in individual if p["group_id"] == g["id"]]
        if not members:
            continue
        a = np.exp(np.mean([np.log(p["matrix"]["values"]) for p in members], axis=0))
        grouped.append({"id": g["id"], "participant_count": len(members), "voting_power": g["voting_power"],
                        "matrix": labeled(a, criteria, criteria), **ahp(a, wd, threshold)})
    powers = np.array([g["voting_power"] for g in grouped], dtype=float)
    if not len(powers) or not np.all(np.isfinite(powers)) or np.any(powers < 0) or powers.sum() <= 0:
        raise ValueError("Active stakeholder voting powers need a positive finite total.")
    powers /= powers.sum()
    for g, power in zip(grouped, powers):
        g["effective_voting_power"] = float(power)
    aggregate = np.exp(np.tensordot(powers, [np.log(g["matrix"]["values"]) for g in grouped], axes=1))
    return {"criterion_ids": criteria, "participants": individual, "groups": grouped,
            "session": {"id": "session", "matrix": labeled(aggregate, criteria, criteria), **ahp(aggregate, wd, threshold)},
            "ri_table": RI, "source": "calculated"}


def topsis(alternatives, criteria, weights):
    ids = [a["id"] for a in alternatives]
    cids = [c["id"] for c in criteria]
    if not ids or not cids:
        raise ValueError("TOPSIS requires alternatives and criteria.")
    x = np.array([[a["values"][c] for c in cids] for a in alternatives], dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(cids),) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, with a positive sum and one entry per criterion.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Decision values must be finite.")
    if any(c["direction"] not in {"max", "min"} for c in criteria):
        raise ValueError("Criterion directions must be max or min.")
    w = w / w.sum()
    norms = np.linalg.norm(x, axis=0)
    r = np.divide(x, norms, out=np.zeros_like(x), where=norms != 0)
    v = r * w
    benefit = np.array([c["direction"] == "max" for c in criteria])
    best = np.where(benefit, v.max(axis=0), v.min(axis=0))
    worst = np.where(benefit, v.min(axis=0), v.max(axis=0))
    db, dw = np.linalg.norm(v - best, axis=1), np.linalg.norm(v - worst, axis=1)
    scores = np.divide(dw, db + dw, out=np.full(len(x), .5), where=(db + dw) != 0)
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
    # Tolerance groups avoid treating floating-point noise as preference changes.
    statistical = np.empty(len(ids))
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and abs(scores[order[start]] - scores[order[end]]) <= TIE_ATOL:
            end += 1
        statistical[order[start:end]] = (start + 1 + end) / 2
        start = end
    return {"decision_matrix": labeled(x, ids, cids), "normalized_matrix": labeled(r, ids, cids),
            "weighted_matrix": labeled(v, ids, cids), "column_norms": dict(zip(cids, norms.tolist())),
            "ideal_best": dict(zip(cids, best.tolist())), "ideal_worst": dict(zip(cids, worst.tolist())),
            "rankings": [{"alternative_id": ids[i], "score": float(scores[i]), "rank": rank,
                          "statistical_rank": float(statistical[i]), "distance_best": float(db[i]), "distance_worst": float(dw[i])}
                         for rank, i in enumerate(order, 1)], "tie_tolerance": TIE_ATOL}


def evaluate(inputs, reused_preferences=None):
    trace = reused_preferences if reused_preferences is not None else preferences(inputs)
    original = dict(zip(trace["criterion_ids"], trace["session"]["weights"]))
    selected = inputs.get("weights") or original
    ids = [c["id"] for c in inputs["criteria"]]
    if inputs.get("weights") is not None and set(selected) != set(ids):
        raise ValueError("Explicit weights must cover exactly the retained criteria.")
    w = np.array([selected[c] for c in ids], dtype=float)
    ranking = topsis(inputs["alternatives"], inputs["criteria"], w)
    return {"preferences": trace, "applied_weights": dict(zip(ids, (w / w.sum()).tolist())), "topsis": ranking}


def compare(current, parent):
    left = {r["alternative_id"]: r for r in parent["topsis"]["rankings"]}
    right = {r["alternative_id"]: r for r in current["topsis"]["rankings"]}
    common = sorted(left.keys() & right.keys())
    # Rerank common alternatives to exclude numerical shifts due to removals.
    a = rankdata([left[k]["statistical_rank"] for k in common])
    b = rankdata([right[k]["statistical_rank"] for k in common])
    reason = "Fewer than two common alternatives or all common alternatives tied."
    defined = len(common) >= 2 and len(set(a)) > 1 and len(set(b)) > 1
    rho = float(spearmanr(a, b).statistic) if defined else None
    tau = float(kendalltau(a, b, variant="b").statistic) if defined else None
    reversals, ties = [], []
    for i, j in combinations(common, 2):
        before = np.sign(left[i]["statistical_rank"] - left[j]["statistical_rank"])
        after = np.sign(right[i]["statistical_rank"] - right[j]["statistical_rank"])
        if before * after < 0:
            reversals.append([i, j])
        elif before != after:
            ties.append([i, j])
    def winners(rows):
        return sorted(k for k, r in rows.items() if r["statistical_rank"] == min(x["statistical_rank"] for x in rows.values()))
    return {"common_alternatives": common, "spearman_rho": rho, "kendall_tau_b": tau,
            "undefined_reason": None if defined else reason, "reversals": reversals,
            "tie_transitions": ties, "winner_changed": winners(left) != winners(right),
            "previous_winners": winners(left), "current_winners": winners(right),
            "score_deltas": {k: right[k]["score"] - left[k]["score"] for k in common},
            "rank_deltas": {k: right[k]["rank"] - left[k]["rank"] for k in common},
            "weight_deltas": {k: v - parent["applied_weights"][k] for k, v in current["applied_weights"].items() if k in parent["applied_weights"]}}

"""Named scientific checks; comparison to PoliInsight is explicitly deferred."""
import json

import numpy as np

REGISTRY = {}
DEFERRED = {"topsis_closeness_and_ranks_match_reference", "computed_results_match_independent_reference_per_variant"}


def check(name):
    def register(fn):
        REGISTRY[name] = fn
        return fn
    return register


@check("all_participant_pairwise_matrices_equal")
@check("all_participants_identical_within_each_variant")
def identical(r, completed):
    matrices = [p["matrix"]["values"] for p in r["preferences"]["participants"]]
    return all(np.allclose(m, matrices[0]) for m in matrices)


@check("all_participant_consistency_ratios")
def participant_cr(r, completed):
    return [p["consistency_ratio"] for p in r["preferences"]["participants"]]


@check("all_group_weight_vectors")
def group_weights(r, completed):
    return [g["weights"] for g in r["preferences"]["groups"]]


@check("session_criterion_weights_in_order_C1_C2_C3_C4")
def session_weights(r, completed):
    return [r["applied_weights"][k] for k in ("C1", "C2", "C3", "C4")]


@check("session_aggregate_pairwise_matrix_all_ones")
def ones(r, completed):
    return bool(np.allclose(r["preferences"]["session"]["matrix"]["values"], 1))


@check("submitted_values_in_saaty_domain")
def valid_ratios(r, completed):
    from src.reference import validate_ratio
    for p in r["inputs"]["participants"]:
        for v in p["judgments"].values():
            validate_ratio(v)
    return True


@check("reciprocal_matrices")
def reciprocal(r, completed):
    pref = r["preferences"]
    return all(np.allclose(np.array(p["matrix"]["values"]) * np.array(p["matrix"]["values"]).T, 1)
               for p in [*pref["participants"], *pref["groups"], pref["session"]])


@check("all_participants_CR_exceeds_warning_threshold_in_each_variant")
@check("inconsistent_inputs_retained_and_flagged")
def flagged(r, completed):
    return len(r["inputs"]["participants"]) == len(r["preferences"]["participants"]) and all(p["flagged"] for p in r["preferences"]["participants"])


@check("max_and_min_variants_have_distinct_aggregate_weights")
def distinct(r, completed):
    # Compare the actual stored outputs of the two configured variants.
    sibling = "S02_MIN" if r["case_key"] == "S02_MAX" else "S02_MAX"
    other = completed.get(sibling)
    if other is None or not other.applied_weights:
        raise ValueError("Sibling variant is unavailable.")
    return not np.allclose(list(r["applied_weights"].values()), list(other.applied_weights.values()))


@check("results_restored")
@check("participation_invariant")
def restored(r, completed):
    c = r["comparison"]
    return bool(c and not c["reversals"] and all(abs(v) < 1e-10 for v in c["score_deltas"].values()) and all(abs(v) < 1e-10 for v in c["weight_deltas"].values()))


@check("dominated_alternative_not_better")
def dominated(r, completed):
    scores = {x["alternative_id"]: x["score"] for x in r["topsis"]["rankings"]}
    return scores["A6"] <= scores["A4"]


@check("inconsistent_participant_count")
def inconsistent_count(r, completed):
    return sum(p["flagged"] for p in r["preferences"]["participants"])



@check("normalized_weights")
def normalized_weights(r, completed):
    weights = list(r["applied_weights"].values())
    return bool(np.isclose(sum(weights), 1) and all(np.isfinite(w) and w >= 0 for w in weights))


@check("bounded_topsis_scores")
def bounded_scores(r, completed):
    scores = [x["score"] for x in r["topsis"]["rankings"]]
    return bool(scores and all(np.isfinite(s) and 0 <= s <= 1 for s in scores))


@check("parent_weights_preserved")
def preserved(r, completed):
    parent = next(x for x in completed.values() if x.run_id == r["parent_run_id"])
    ids = list(r["applied_weights"])
    weights = np.array([parent.applied_weights[k] for k in ids])
    return bool(np.allclose(list(r["applied_weights"].values()), weights / weights.sum(), rtol=0, atol=1e-10))

def validate_assertions(specs):
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Each assertion must be an object.")
        if spec.get("check") not in REGISTRY.keys() | DEFERRED:
            raise ValueError(f"Unknown assertion: {spec.get('check')!r}")
        if spec.get("check") in DEFERRED and spec.get("deferred") is not True:
            raise ValueError("External comparison assertions must explicitly set deferred: true.")
        if "id" not in spec or "expected" not in spec:
            raise ValueError("Assertions require id, check, expected.")
        try:
            json.dumps(spec["expected"], allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Assertion expectations must be finite JSON data.") from error
        tolerance = spec.get("tolerance", 1e-8)
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Assertion tolerance must be finite and nonnegative.")


def evaluate_assertions(specs, result, completed, pending_sibling=False):
    output = []
    for spec in specs:
        name = spec["check"]
        record = {**spec, "tolerance": spec.get("tolerance", 1e-8)}
        if name in DEFERRED:
            record.update(status="deferred", observed=None, reason="PoliInsight comparison is outside this reference-only release.")
        elif pending_sibling and name == "max_and_min_variants_have_distinct_aggregate_weights":
            record.update(status="pending", observed=None)
        else:
            try:
                observed = REGISTRY[name](result, completed)
                expected = spec["expected"]
                passed = bool(np.allclose(observed, expected, rtol=0, atol=record["tolerance"]))
                record.update(status="passed" if passed else "failed", observed=observed)
            except Exception as error:
                record.update(status="error", observed=None, reason=str(error))
        output.append(record)
    return output

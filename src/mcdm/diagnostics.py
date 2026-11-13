"""Deterministic DCL diagnostics"""
from copy import deepcopy
from itertools import combinations

import numpy as np

from src.mcdm.reference import compare, evaluate, topsis

def ranking(inputs, weights):
    """Compute a TOPSIS ranking for the current alternatives using the supplied criterion weights."""
    cids = [c["id"] for c in inputs["criteria"]]
    values = np.array([weights[c] for c in cids], dtype=float)
    return {
        "applied_weights": dict(zip(cids, (values / values.sum()).tolist())),
        "topsis": topsis(inputs["alternatives"], inputs["criteria"], values)
    }

def delta(current, baseline):
    """Compare a current ranking against a baseline and summarize rank, winner, and score deltas."""
    return compare(current, baseline)

def robustness(inputs, baseline):
    """Stress-test a baseline run by perturbing weights and removing criteria to assess sensitivity."""
    rows = []
    weights = baseline["applied_weights"]

    for cid, weight in weights.items():
        for factor in (.8, .9, 1.1, 1.2):
            row = {
                "criterion_id": cid,
                "factor": factor,
                "operation": "weight_perturbation"
            }
            target = weight * factor
            if target > 1 or weight >= 1 or weight == 0:
                row.update(status="not_evaluable", reason="Cannot proportionally rescale this boundary weight.")
            else:
                changed = {key: target if key == cid else value * (1-target)/(1-weight) for key, value in weights.items()}
                outcome = ranking(inputs, changed)
                row.update(status="evaluated", weights=outcome["applied_weights"], comparison=delta(outcome, baseline))
            rows.append(row)
        row = {"criterion_id": cid, "operation": "criterion_removal"}
        reduced = deepcopy(inputs)
        reduced["criteria"] = [c for c in inputs["criteria"] if c["id"] != cid]
        if not reduced["criteria"] or sum(v for key, v in weights.items() if key != cid) <= 0:
            row.update(status="not_evaluable", reason="No positive retained criterion weight.")
        else:
            outcome = ranking(reduced, weights)
            row.update(status="evaluated", weights=outcome["applied_weights"], comparison=delta(outcome, baseline))
        rows.append(row)
    return rows

def compute_diagnostics(inputs, trace):
    """Build detailed group-level, cross-group, and session-level diagnostics from a decision trace."""
    pref = trace["preferences"]
    cids = pref["criterion_ids"]
    active = [c["id"] for c in inputs["criteria"]]
    if not set(active) <= set(cids):
        raise ValueError("Applied criteria are absent from preference evidence.")
    group_results = {}
    participants_by_group = {}
    for group in inputs["groups"]:
        members = [p for p in pref["participants"] if p["group_id"] == group["id"]]
        participants_by_group[group["id"]] = members
    for group in pref["groups"]:
        gid = group["id"]
        outcome = ranking(inputs, dict(zip(cids, group["weights"])))
        members = participants_by_group[gid]
        member_outcomes = [ranking(inputs, dict(zip(cids, p["weights"]))) for p in members]
        values = np.asarray([p["weights"] for p in members])
        group_results[gid] = {
            "rankings": outcome["topsis"]["rankings"],
            "applied_weights": outcome["applied_weights"],
            "variation": {cid: {"min": float(values[:, i].min()), "max": float(values[:, i].max()),
                                "population_std": float(values[:, i].std(ddof=0))} for i, cid in enumerate(cids)},
            "participant_agreement": [delta(a, b) for a, b in combinations(member_outcomes, 2)],
            "agreement_coverage": {"participants": len(members), "pairs": len(members)*(len(members)-1)//2,
                                   "limitation": "Describes modeled submissions, not population agreement; singleton groups provide no pairwise agreement evidence."},
            "robustness": robustness(inputs, outcome),
        }
    cross = []
    for a, b in combinations(group_results, 2):
        left, right = group_results[a], group_results[b]
        cross.append({"group_ids": [a, b], "comparison": delta(
            {"topsis": {"rankings": left["rankings"]}, "applied_weights": left["applied_weights"]},
            {"topsis": {"rankings": right["rankings"]}, "applied_weights": right["applied_weights"]})})
    influence = {"groups": {}, "participants_by_group": {}}
    for group in inputs["groups"]:
        gid = group["id"]
        omissions = [p for p in inputs["participants"] if p["group_id"] == gid]
        participant_rows = []
        for omitted in [None, *omissions]:
            altered = deepcopy(inputs)
            altered["participants"] = [p for p in inputs["participants"] if
                (p["group_id"] != gid if omitted is None else p["id"] != omitted["id"])]
            try:
                # Recompute preferences after omission, but honor the run's explicit/reused applied weights.
                result = evaluate(altered)
                row = {"status": "evaluated", "comparison": delta(result, trace)}
            except (ValueError, FloatingPointError) as error:
                row = {"status": "not_evaluable", "reason": str(error)}
            if omitted is None:
                if not omissions:
                    row = {"status": "not_evaluable", "reason": "Group has no observed participants."}
                influence["groups"][gid] = row
            else:
                participant_rows.append(row)
        # Cases retain distribution/minority effects but never participant IDs or names.
        influence["participants_by_group"][gid] = {"omission_count": len(omissions), "cases": participant_rows}
    return {"schema_version": 1, "groups": group_results, "cross_group_agreement": cross,
            "influence": influence, "session_robustness": robustness(inputs, trace),
            "limitations": ["Counterfactual omissions do not establish causal influence.",
                            "Robustness covers only the recorded perturbation factors and single-criterion omissions.",
                            "Fixed applied weights are preserved during omissions; preference changes may therefore leave rankings unchanged."]}


def diagnostic_summaries(diagnostics):
    """Condense detailed diagnostic evidence into compact summaries while preserving key changes."""
    def compact_comparison(comparison):
        """Reduce a full comparison result to the fields most relevant for human-readable summaries."""
        fields = ("spearman_rho", "kendall_tau_b", "undefined_reason", "winner_changed",
                  "previous_winners", "current_winners", "reversals", "tie_transitions")
        return {**{key: comparison[key] for key in fields},
                "nonzero_rank_deltas": {key: value for key, value in comparison["rank_deltas"].items() if value != 0},
                "maximum_absolute_score_delta": max((abs(v) for v in comparison["score_deltas"].values()), default=0.)}
    
    def cases(rows):
        """Convert raw diagnostic rows into compact summary entries while preserving notable comparisons."""
        return [{**{key: value for key, value in row.items() if key not in {"comparison", "weights"}},
                 **({"comparison": compact_comparison(row["comparison"])} if "comparison" in row else {})}
                for row in rows]
    def agreement(rows):
        """Summarize the distribution of pairwise agreement statistics across a set of comparisons."""
        # Preserve all correlation values and their multiplicities, including undefined cases.
        distribution = {}
        for row in rows:
            key = (row["spearman_rho"], row["kendall_tau_b"], row["undefined_reason"])
            distribution[key] = distribution.get(key, 0)+1
        return {"pair_count": len(rows), "correlation_distribution": [
            {"spearman_rho": key[0], "kendall_tau_b": key[1], "undefined_reason": key[2], "pair_count": count}
            for key, count in distribution.items()]}
    def robustness_findings(rows):
        """Highlight the evaluated perturbation cases and the subset that changed winners or rankings."""
        evaluated = [row for row in rows if "comparison" in row]
        def condition(row):
            return {key: row[key] for key in ("operation", "criterion_id", "factor", "status", "reason") if key in row}
        return [{"tested_conditions": [condition(row) for row in rows],
            "evaluated_count": len(evaluated),
            "not_evaluable": [condition(row) for row in rows if "comparison" not in row],
            "winner_change_cases": [{**condition(row), "comparison": compact_comparison(row["comparison"])}
                for row in evaluated if row["comparison"]["winner_changed"]],
            "other_rank_change_cases": [{**condition(row), "comparison": compact_comparison(row["comparison"])}
                for row in evaluated if not row["comparison"]["winner_changed"] and
                    (row["comparison"]["tie_transitions"] or any(row["comparison"]["rank_deltas"].values()))],
            "maximum_absolute_score_delta": max((abs(value) for row in evaluated for value in row["comparison"]["score_deltas"].values()), default=0.)}]
    return {
        "groups": {gid: {"participant_agreement": agreement(row["participant_agreement"]),
                         "robustness": cases(row["robustness"]),
                         "robustness_findings": robustness_findings(row["robustness"])} for gid, row in diagnostics["groups"].items()},
        "cross_group_agreement": [{"group_ids": row["group_ids"], "comparison": compact_comparison(row["comparison"])}
                                  for row in diagnostics["cross_group_agreement"]],
        "group_influence": {gid: cases([row])[0] for gid, row in diagnostics["influence"]["groups"].items()},
        "participant_influence": {gid: {"omission_count": row["omission_count"], "cases": cases(row["cases"])}
                                  for gid, row in diagnostics["influence"]["participants_by_group"].items()},
        "session_robustness": cases(diagnostics["session_robustness"]),
        "session_robustness_findings": robustness_findings(diagnostics["session_robustness"]),
        "coverage": "Every tested intervention and pairwise correlation outcome is retained. Full score/weight delta vectors remain in the saved diagnostic evidence; summaries retain maximum absolute score movement and nonzero rank deltas.",
    }

import math

import numpy as np
from scipy.stats import kendalltau, spearmanr

from src.mcdm.diagnostics import (
    TIE_ATOL,
    statistical_ranks,
    winner_ids,
)


SUPPORTED_METRICS = {
    "weight_delta",
    "score_delta",
    "winner_changed",
    "spearman_rho",
    "kendall_tau_b",
}


def _session_values(result):
    if result.execution.get("numerical_status") != "completed":
        raise ValueError(
            f"{result.case_key}: numerical results are unavailable."
        )

    # A run may have failed later assertions while retaining valid
    # numerical results. Validate those results independently.
    result.validate_numerical_complete()

    calculation = result.mcdm.session

    scores = {
        item.alternative_id: float(item.closeness_coefficient)
        for item in calculation.ranking.rankings
    }

    weights = dict(
        calculation.metadata.applied_criterion_weights
    )

    return scores, weights


def _align_ids(current_ids, reference_ids, policy, label):
    current_ids = set(current_ids)
    reference_ids = set(reference_ids)

    if policy == "require_identical_ids":
        if current_ids != reference_ids:
            raise ValueError(
                f"Comparison requires identical {label} IDs."
            )

    elif policy != "common_ids":
        raise ValueError(
            f"Unsupported {label} alignment policy: {policy!r}."
        )

    return {
        "common": sorted(current_ids & reference_ids),
        "current_only": sorted(current_ids - reference_ids),
        "reference_only": sorted(reference_ids - current_ids),
    }


def compare_results(current, reference, specification):
    identifier = specification.get("id")

    if not isinstance(identifier, str) or not identifier:
        raise ValueError(
            "A comparison requires a nonempty ID."
        )

    if specification.get("reference") != reference.case_key:
        raise ValueError(
            "Comparison reference does not match the supplied result."
        )

    requested = specification.get("metrics")

    if (
        not isinstance(requested, list)
        or not requested
        or any(not isinstance(item, str) for item in requested)
        or len(requested) != len(set(requested))
        or set(requested) - SUPPORTED_METRICS
    ):
        raise ValueError(
            "Comparison metrics must be unique supported names."
        )

    tie_atol = specification.get("tie_atol", TIE_ATOL)

    if (
        isinstance(tie_atol, bool)
        or not isinstance(tie_atol, (int, float))
        or not math.isfinite(tie_atol)
        or tie_atol < 0
    ):
        raise ValueError(
            "tie_atol must be finite and nonnegative."
        )

    rerank_common = specification.get(
        "rerank_common_alternatives",
        False,
    )

    if not isinstance(rerank_common, bool):
        raise ValueError(
            "rerank_common_alternatives must be a Boolean."
        )

    current_scores, current_weights = _session_values(current)
    reference_scores, reference_weights = _session_values(reference)

    alternative_policy = specification.get(
        "alternative_alignment",
        "require_identical_ids",
    )
    criterion_policy = specification.get(
        "criterion_alignment",
        "require_identical_ids",
    )

    alternatives = _align_ids(
        current_scores,
        reference_scores,
        alternative_policy,
        "alternative",
    )
    criteria = _align_ids(
        current_weights,
        reference_weights,
        criterion_policy,
        "criterion",
    )

    common_alternatives = alternatives["common"]
    common_criteria = criteria["common"]

    metrics = {}
    undefined_reasons = {}

    if "weight_delta" in requested:
        if common_criteria:
            # Preserve actual applied weights. Do not renormalize
            # their intersection merely to perform a comparison.
            metrics["weight_delta"] = {
                criterion_id: (
                    current_weights[criterion_id]
                    - reference_weights[criterion_id]
                )
                for criterion_id in common_criteria
            }

        else:
            metrics["weight_delta"] = None
            undefined_reasons["weight_delta"] = (
                "No common ranking criteria."
            )

    if "score_delta" in requested:
        if common_alternatives:
            metrics["score_delta"] = {
                alternative_id: (
                    current_scores[alternative_id]
                    - reference_scores[alternative_id]
                )
                for alternative_id in common_alternatives
            }

        else:
            metrics["score_delta"] = None
            undefined_reasons["score_delta"] = (
                "No common alternatives."
            )

    current_winners = winner_ids(
        current_scores,
        tie_atol=tie_atol,
    )
    reference_winners = winner_ids(
        reference_scores,
        tie_atol=tie_atol,
    )

    if "winner_changed" in requested:
        # Compare the actual full winner sets, including alternatives
        # that occur in only one result.
        metrics["winner_changed"] = (
            current_winners != reference_winners
        )

    correlation_metrics = {
        "spearman_rho",
        "kendall_tau_b",
    } & set(requested)

    rank_evidence = None

    if correlation_metrics:
        if rerank_common:
            current_rank_scores = {
                alternative_id: current_scores[alternative_id]
                for alternative_id in common_alternatives
            }
            reference_rank_scores = {
                alternative_id: reference_scores[alternative_id]
                for alternative_id in common_alternatives
            }

        else:
            current_rank_scores = current_scores
            reference_rank_scores = reference_scores

        current_ranks = statistical_ranks(
            current_rank_scores,
            tie_atol=tie_atol,
        )
        reference_ranks = statistical_ranks(
            reference_rank_scores,
            tie_atol=tie_atol,
        )

        x = np.asarray([
            current_ranks[alternative_id]
            for alternative_id in common_alternatives
        ])
        y = np.asarray([
            reference_ranks[alternative_id]
            for alternative_id in common_alternatives
        ])

        rank_evidence = {
            "alternative_ids": common_alternatives,
            "current_ranks": x.tolist(),
            "reference_ranks": y.tolist(),
        }

        if len(common_alternatives) < 2:
            reason = "Fewer than two common alternatives."

        elif len(set(x)) < 2 or len(set(y)) < 2:
            reason = (
                "At least one aligned ranking contains only ties."
            )

        else:
            reason = None

        for metric in sorted(correlation_metrics):
            if reason is not None:
                metrics[metric] = None
                undefined_reasons[metric] = reason
                continue

            if metric == "spearman_rho":
                coefficient = float(
                    spearmanr(x, y).statistic
                )

            else:
                coefficient = float(
                    kendalltau(x, y, variant="b").statistic
                )

            if math.isfinite(coefficient):
                metrics[metric] = coefficient

            else:
                metrics[metric] = None
                undefined_reasons[metric] = (
                    "The correlation calculation was undefined."
                )

    return {
        "schema_version": "1.0",
        "id": identifier,
        "status": "completed",
        "current_run_id": current.run_id,
        "reference_run_id": reference.run_id,
        "current_case_key": current.case_key,
        "reference_case_key": reference.case_key,
        "scope": "session",
        "delta_direction": "current_minus_reference",
        "tie_atol": tie_atol,
        "rerank_common_alternatives": rerank_common,
        "alternative_alignment": {
            "policy": alternative_policy,
            **alternatives,
        },
        "criterion_alignment": {
            "policy": criterion_policy,
            **criteria,
        },
        "current_winner_ids": current_winners,
        "reference_winner_ids": reference_winners,
        "winner_scope": "full_results",
        "rank_evidence": rank_evidence,
        "metrics": metrics,
        "undefined_reasons": undefined_reasons,
    }


def evaluate_comparisons(run, result, completed):
    records = []

    for specification in run.experiment.comparisons:
        try:
            reference_case = specification["reference"]

            reference_execution_key = run.reference_runs[
                reference_case
            ]

            reference = completed[reference_execution_key]

            record = compare_results(
                result,
                reference,
                specification,
            )

        except Exception as error:
            record = {
                "id": specification.get("id"),
                "status": "error",
                "reference_case_key": specification.get(
                    "reference"
                ),
                "reason": (
                    f"{type(error).__name__}: {error}"
                ),
            }

        records.append(record)

    return records


def comparison_view(result, *, tie_atol, alternative_ids=None):
    rows = [
        row
        for row in result.mcdm.session.ranking.rankings
        if alternative_ids is None
        or row.alternative_id in alternative_ids
    ]

    scores = {
        row.alternative_id: float(row.closeness_coefficient)
        for row in rows
    }
    ranks = statistical_ranks(
        scores,
        tie_atol=tie_atol,
    )
    rows = sorted(
        rows,
        key=lambda row: (
            -row.closeness_coefficient,
            row.alternative_id,
        ),
    )

    return {
        "applied_weights": dict(
            result.mcdm.session.metadata.applied_criterion_weights
        ),
        "topsis": {
            "rankings": [
                {
                    "alternative_id": row.alternative_id,
                    "score": row.closeness_coefficient,
                    "rank": index,
                    "statistical_rank": ranks[
                        row.alternative_id
                    ],
                }
                for index, row in enumerate(rows, start=1)
            ]
        },
    }

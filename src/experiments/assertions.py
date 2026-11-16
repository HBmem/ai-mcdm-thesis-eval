"""Validate assertion declarations; evaluation against results is a separate step."""
import json
import math


# Accepted parameter sets for the checks declared by S01, S05, and S06. This registry
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
}


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

"""Read-only viewer for independently generated evaluation artifacts."""
import argparse
import json
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
import streamlit as st

from src.results import RunResult, safe_artifact

ROOT = Path(__file__).resolve().parent


@st.cache_data
def decode_result(payload: bytes) -> dict:
    try:
        return RunResult.model_validate_json(payload).model_dump()
    except (TypeError, AttributeError, KeyError) as error:
        raise ValueError(f"Malformed result structure: {error}") from error


def matrix_view(label, matrix):
    st.caption(label)
    st.dataframe(pd.DataFrame(matrix["values"], index=matrix["row_ids"], columns=matrix["column_ids"]), width="stretch")


def read_manifest(path):
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or value.get("schema_version") != "1.0" or value.get("status") not in {"running", "completed"} or not isinstance(value.get("runs"), list):
        raise ValueError("Unsupported or invalid batch manifest.")
    for entry in value["runs"]:
        if not isinstance(entry, dict) or not all(k in entry for k in ("experiment_id", "case_key", "file", "status")):
            raise ValueError("Invalid run entry in batch manifest.")
        if not all(isinstance(entry[k], str) for k in ("experiment_id", "case_key", "file", "status")):
            raise ValueError("Manifest run identifiers, filename, and status must be strings.")
        safe_artifact(path.parent, entry["file"])
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args, _ = parser.parse_known_args(argv)
    results_root = args.results.resolve()
    st.set_page_config(page_title="MCDM evaluations", layout="wide")
    st.title("MCDM evaluations")
    st.caption("Independent deterministic reference results. PoliInsight comparison and AI interpretation evaluation are deferred.")
    batches = []
    if results_root.is_dir():
        for path in sorted(results_root.glob("*/manifest.json"), reverse=True):
            if path.resolve().parent.parent != results_root:
                continue
            try:
                batches.append((path.parent, read_manifest(path)))
            except (OSError, ValueError, TypeError, KeyError) as error:
                st.warning(f"Cannot read batch {path.parent.name}: {error}")
    if not batches:
        st.info("No readable result batches. Run python run_experiment.py run to generate artifacts.")
        return
    selected = st.sidebar.selectbox("Batch", range(len(batches)), format_func=lambda i: batches[i][0].name)
    batch, manifest = batches[selected]
    st.caption(f"Batch status: {manifest['status']}")
    entries = manifest["runs"]
    if not entries:
        st.info("This batch has no saved runs yet.")
        return
    st.sidebar.caption("Filters select saved results; they do not execute calculations.")
    scenario = st.sidebar.selectbox("Scenario", sorted({e["experiment_id"] for e in entries}))
    variants = [e for e in entries if e["experiment_id"] == scenario]
    entry = st.sidebar.selectbox("Variant", variants, format_func=lambda e: e["case_key"])
    try:
        payload = safe_artifact(batch, entry["file"]).read_bytes()
        result = decode_result(payload)
    except (OSError, ValueError, ValidationError) as error:
        st.error(f"Cannot read result: {error}")
        return
    st.subheader(result["name"])
    st.caption(f"Run: {result['run_id']} · Case: {result['case_key']}")
    overview, variables, calculations, comparisons, exports = st.tabs(
        ["Overview", "Variables", "Calculations", "Comparisons", "Provenance/export"]
    )
    with overview:
        st.metric("Status", result["status"])
        if result["execution"].get("description"):
            st.write(result["execution"]["description"])
        for message in result["execution"].get("messages", []):
            st.error(message)
        if result["topsis"]:
            winner = result["topsis"]["rankings"][0]
            st.metric("First alternative in presentation order", winner["alternative_id"])
            st.metric("Session consistency ratio", f"{result['preferences']['session']['consistency_ratio']:.6f}")
            st.dataframe(pd.DataFrame(result["topsis"]["rankings"]), width="stretch")
        st.caption("Assertion outcomes; deferred checks have not validated PoliInsight.")
        if result["assertions"]:
            st.dataframe(pd.DataFrame([{k: json.dumps(v) if k in {"expected", "observed"} or isinstance(v, (dict, list)) else v for k, v in a.items()} for a in result["assertions"]]), width="stretch")
        else:
            st.info("No assertions recorded for this run.")
    with variables:
        x = result["inputs"]
        if not x:
            st.info("Resolved inputs are unavailable for this run.")
        else:
            if x.get("scenario"):
                st.write(x["scenario"]["policy_question"])
                st.caption(x["research_purpose"])
            st.write("Criteria")
            st.dataframe(pd.DataFrame(x["criteria"]), width="stretch")
            st.write("Processing")
            st.json(x["processing"])
            st.write("Stakeholder groups")
            groups = []
            effective = {g["id"]: g["effective_voting_power"] for g in result["preferences"].get("groups", [])}
            for g in x["groups"]:
                groups.append({**g, "participant_count": sum(p["group_id"] == g["id"] for p in x["participants"]), "effective_voting_power": effective.get(g["id"], 0.)})
            st.dataframe(pd.DataFrame(groups), width="stretch")
            st.write("Participant judgments (numeric ratios)")
            st.dataframe(pd.DataFrame([{"id": p["id"], "group_id": p["group_id"], **p["judgments"]} for p in x["participants"]]), width="stretch")
            if result["topsis"]:
                matrix_view("Decision matrix", result["topsis"]["decision_matrix"])
            with st.expander("Resolved input and variation"):
                st.json(x)
    with calculations:
        pref = result["preferences"]
        if not pref:
            st.info("Calculations are unavailable for this run.")
        else:
            st.caption(f"Preference trace: {pref['source']}. Applied TOPSIS weights may be restricted or overridden.")
            st.dataframe(pd.DataFrame([{"criterion_id": k, "weight": v} for k, v in result["applied_weights"].items()]))
            rows = []
            for level, items in (("participant", pref["participants"]), ("group", pref["groups"]), ("session", [pref["session"]])):
                for item in items:
                    rows.append({"level": level, "id": item["id"], **{key: item[key] for key in ("wd", "consistency_ratio", "flagged", "threshold", "lambda_max", "consistency_convention")}})
                    with st.expander(f"{level}: {item['id']}"):
                        matrix_view("Pairwise comparison matrix", item["matrix"])
                        st.dataframe(pd.DataFrame({"criterion_id": pref["criterion_ids"], "weight": item["weights"]}))
            st.write("Consistency diagnostics")
            st.dataframe(pd.DataFrame(rows), width="stretch")
            if x.get("target_profiles"):
                target_rows = []
                for g in pref["groups"]:
                    if g["id"] in x["target_profiles"]:
                        for cid, target, achieved in zip(pref["criterion_ids"], x["target_profiles"][g["id"]], g["weights"]):
                            target_rows.append({"group_id": g["id"], "criterion_id": cid, "target": target, "achieved": achieved, "difference": achieved-target})
                st.write("Target and achieved profiles")
                st.dataframe(pd.DataFrame(target_rows))
            for key in ("normalized_matrix", "weighted_matrix"):
                matrix_view(key.replace("_", " ").title(), result["topsis"][key])
            with st.expander("TOPSIS ideals, distances, and full details"):
                st.json(result["topsis"])
    with comparisons:
        comparison = result["comparison"]
        if comparison is None:
            st.info("This run has no baseline comparison.")
        else:
            st.caption(f"Reference: {comparison['reference_case_key']}")
            st.json(comparison)
            if comparison["score_deltas"]:
                st.bar_chart(pd.DataFrame.from_dict(comparison["score_deltas"], orient="index", columns=["score_change"]))
        if scenario == "S08":
            sensitivity = []
            intervals = []
            for item in variants:
                try:
                    saved = decode_result(safe_artifact(batch, item["file"]).read_bytes())
                    variation = saved["inputs"].get("variation", {})
                    for row in saved["topsis"].get("rankings", []):
                        sensitivity.append({"criterion": variation["sensitivity_criterion"], "weight": variation["sensitivity_weight"], "alternative": row["alternative_id"], "score": row["score"]})
                    if saved.get("comparison") and saved["comparison"].get("winner_change_interval"):
                        intervals.append({"criterion": variation["sensitivity_criterion"], "interval": saved["comparison"]["winner_change_interval"]})
                except (OSError, ValueError, KeyError):
                    st.warning(f"Sensitivity data unavailable: {item['case_key']}")
            if sensitivity:
                frame = pd.DataFrame(sensitivity)
                criterion = st.selectbox("Sensitivity criterion", sorted(frame["criterion"].unique()))
                st.line_chart(frame[frame["criterion"] == criterion].pivot(index="weight", columns="alternative", values="score"))
                st.caption("Winner changes are bounded by tested weight intervals; exact thresholds were not calculated.")
                st.dataframe(pd.DataFrame(intervals))
    with exports:
        st.json(result["provenance"])
        st.download_button("Download run JSON", payload, file_name=f"{result['run_id']}.json", mime="application/json")
        for filename in ("summary.csv", "rankings.csv", "weights.csv", "assertions.csv"):
            path = batch / filename
            if path.is_file() and path.resolve().parent == batch.resolve():
                st.download_button(f"Download {filename}", path.read_bytes(), file_name=filename, mime="text/csv")


if __name__ == "__main__":
    main()

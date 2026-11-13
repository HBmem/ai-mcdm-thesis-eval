"""Isolated, read-only dashboard concept. Run from this directory for its theme."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import decode_result, read_manifest  # Reuse validation without running main.
from src.results import safe_artifact


@st.cache_data(max_entries=256, show_spinner=False)
def load_run(payload):
    return decode_result(payload)


def table(rows, label, **kwargs):
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", alt=label, **kwargs)


def matrix(trace, label):
    st.dataframe(
        pd.DataFrame(trace["values"], index=trace["row_ids"], columns=trace["column_ids"]),
        alt=label,
    )


def bar_chart(frame, category, value, title, percent=False):
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y(f"{category}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=280)),
        x=alt.X(f"{value}:Q", title=title, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%" if percent else ".1f")),
        color=alt.Color(f"{category}:N", legend=None),
        tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{value}:Q", format=".4f")],
    ).properties(height=250)
    st.altair_chart(chart, width="stretch", alt=title)


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--results", type=Path, default=ROOT / "results")
args, _ = parser.parse_known_args()
st.set_page_config(page_title="Decision lab · dashboard demo", page_icon=":material/analytics:", layout="wide")
st.caption("DECISION LAB / INDEPENDENT MCDM EVALUATION")
st.title("From preferences to priorities")
st.caption("Dashboard demo · Saved reference results · Synthetic transportation scenarios")

batches = []
for path in sorted(args.results.resolve().glob("*/manifest.json"), reverse=True):
    if path.resolve().parent.parent != args.results.resolve():
        continue
    try:
        batches.append((path.parent, read_manifest(path)))
    except (OSError, ValueError, TypeError, KeyError) as error:
        st.warning(f"Cannot read batch {path.parent.name}: {error}")
if not batches:
    st.info("No readable saved batches. Select a results directory with --results. This demo never executes experiments.")
    st.stop()

with st.sidebar:
    st.header("Explore results", icon=":material/tune:")
    batch_index = st.selectbox("Saved batch", range(len(batches)), format_func=lambda i: batches[i][0].name)
    batch, manifest = batches[batch_index]
    if st.button("Refresh saved artifacts", icon=":material/refresh:", width="stretch"):
        st.rerun()
    st.caption(f"Batch: {manifest['status']} · {len(manifest['runs'])} saved runs")
    st.caption("Filters only read saved files. No calculations are executed.")

entries = manifest["runs"]
if not entries:
    st.info("This batch has no saved runs yet. Refresh after artifacts are saved.")
    st.stop()
with st.sidebar:
    scenario = st.selectbox("Scenario", sorted({e["experiment_id"] for e in entries}), key="scenario")
    variants = [e for e in entries if e["experiment_id"] == scenario]
    case = st.selectbox("Saved variant", [e["case_key"] for e in variants], key="case")
entry = next(e for e in variants if e["case_key"] == case)
try:
    payload = safe_artifact(batch, entry["file"]).read_bytes()
    result = load_run(payload)
except (OSError, ValueError, TypeError, KeyError) as error:
    st.error(f"Cannot read saved result: {error}")
    st.stop()

inputs = result["inputs"]
preferences = result["preferences"]
rankings = result["topsis"].get("rankings", [])
names = {a["id"]: a["name"] for a in inputs.get("alternatives", [])}
criteria = {c["id"]: c["name"] for c in inputs.get("criteria", [])}
checks = Counter(a["status"] for a in result["assertions"])
session = preferences.get("session", {})
st.subheader(f"{case} · {result['name']}")
if inputs.get("research_purpose"):
    st.write(inputs["research_purpose"])
with st.container(horizontal=True):
    st.badge(f"Reference run: {result['status']}", color="green" if result["status"] == "passed" else "red")
    st.badge(f"{checks['deferred']} deferred checks", color="orange")
    st.badge(f"{len(inputs.get('participants', []))} participants", color="gray")
st.caption("Reference checks do not validate PoliInsight. External comparison and AI interpretation evaluation are deferred.")
for message in result["execution"].get("messages", []):
    st.error(message)

if rankings:
    ordered = sorted(rankings, key=lambda row: -row["score"])
    top = ordered[0]["score"]
    winners = [r for r in ordered if abs(r["score"] - top) <= 1e-12]
    gap = top - ordered[1]["score"] if len(ordered) > 1 else None
    cols = st.columns(4)
    cols[0].metric("Leading alternative" if len(winners) == 1 else "Tied leaders", ", ".join(r["alternative_id"] for r in winners), border=True)
    cols[1].metric("Top TOPSIS score", f"{top:.4f}", border=True, help="Relative closeness to the ideal solution, on a 0–1 scale. This is not a probability.")
    cols[2].metric("Gap to second place", f"{gap:.4f}" if gap is not None else "N/A", border=True, help="An absolute score difference, not a measure of statistical confidence.")
    cols[3].metric("Session consistency ratio", f"{session['consistency_ratio']:.4f}" if session else "N/A", border=True, help="Calculated from the saved preference trace; applied ranking weights may differ.")
    st.markdown("**Leading option" + ("s" if len(winners) > 1 else "") + ":** " + "; ".join(names.get(r["alternative_id"], r["alternative_id"]) for r in winners))
if session.get("flagged"):
    st.warning(f"Session CR exceeds its saved threshold of {session['threshold']:.2f}. Judgments were retained.")

view = st.segmented_control("View", ["Decision overview", "Validation", "Sensitivity & comparison", "Calculation trace", "Export"], default="Decision overview", key="view")
if view == "Decision overview":
    if not rankings:
        st.info("No rankings were saved for this run. Review validation and execution messages.")
    else:
        left, right = st.columns([3, 2])
        ranking_frame = pd.DataFrame(rankings)
        ranking_frame["Alternative"] = ranking_frame["alternative_id"].map(lambda x: f"{x} · {names.get(x, x)}")
        with left.container(border=True):
            st.subheader("How the options rank")
            st.caption("Higher TOPSIS closeness is preferred. Ties use a score tolerance of 1e-12.")
            bar_chart(ranking_frame, "Alternative", "score", "TOPSIS closeness · 0–1")
        with right.container(border=True):
            st.subheader("What drives the ranking")
            st.caption("Applied weights, including any saved restriction or override.")
            weights = pd.DataFrame([{"Criterion": criteria.get(k, k), "weight": v} for k, v in result["applied_weights"].items()])
            bar_chart(weights, "Criterion", "weight", "Share of applied weight", percent=True)
        with st.container(border=True):
            st.subheader("Ranking details")
            st.dataframe(ranking_frame[["rank", "Alternative", "score", "statistical_rank"]], hide_index=True, alt="Saved ranks and closeness scores", column_config={"score": st.column_config.ProgressColumn("TOPSIS score", min_value=0, max_value=1, format="%.4f"), "rank": "Presentation rank", "statistical_rank": "Rank with ties"})
    with st.expander("Scenario inputs and criterion directions"):
        st.write(inputs.get("scenario", {}).get("policy_question", "No policy question saved."))
        table(inputs.get("criteria", []), "Criteria, measurement units and benefit or cost direction")
        if result["topsis"]:
            matrix(result["topsis"]["decision_matrix"], "Raw decision matrix")
        st.json(inputs)

elif view == "Validation":
    st.subheader("Checks and judgment quality")
    for col, status in zip(st.columns(4), ["passed", "failed", "deferred", "blocked"]):
        col.metric(f"{status.capitalize()} checks", checks[status], border=True)
    status_filter = st.multiselect("Assertion status", sorted(checks), default=sorted(checks))
    assertions = [{**a, "check": a["check"].replace("_", " "), "expected": json.dumps(a.get("expected")), "observed": json.dumps(a.get("observed"))} for a in result["assertions"] if a["status"] in status_filter]
    if assertions:
        table(assertions, "Assertion outcomes, including deferred and failed checks")
    else:
        st.info("No assertions match this selection.")
    st.caption("Deferred checks have not been evaluated and are excluded from passed counts.")
    diagnostics = [{"Level": level, "ID": item["id"], "CR": item["consistency_ratio"], "Threshold": item["threshold"], "Flagged": item["flagged"]} for level, items in [("Participant", preferences.get("participants", [])), ("Group", preferences.get("groups", [])), ("Session", [session] if session else [])] for item in items]
    if diagnostics:
        st.subheader("Consistency by aggregation level")
        table(diagnostics, "Participant, group and session consistency diagnostics")

elif view == "Sensitivity & comparison":
    comparison = result["comparison"]
    st.subheader("Change from the saved reference")
    if comparison:
        st.caption(f"Reference case: {comparison['reference_case_key']}")
        st.info("The leading set changed." if comparison.get("winner_changed") else "The leading set is unchanged.")
        deltas = comparison.get("score_deltas", {})
        if deltas:
            frame = pd.DataFrame([{"Alternative": names.get(k, k), "Score change": v} for k, v in deltas.items()])
            st.altair_chart(alt.Chart(frame).mark_bar().encode(y=alt.Y("Alternative:N", sort="-x"), x="Score change:Q", color=alt.condition(alt.datum["Score change"] >= 0, alt.value("#0f766e"), alt.value("#b45309")), tooltip=["Alternative", alt.Tooltip("Score change:Q", format=".6f")]), width="stretch", alt="Score changes for alternatives common to the saved reference and current run")
        with st.expander("Full saved comparison"):
            st.json(comparison)
    else:
        st.info("This run has no saved baseline comparison.")
    if scenario == "S08":
        points, intervals = [], []
        for item in variants:
            try:
                saved = load_run(safe_artifact(batch, item["file"]).read_bytes())
                variation = saved["inputs"].get("variation", {})
                for row in saved["topsis"].get("rankings", []):
                    points.append({"Criterion": variation["sensitivity_criterion"], "Weight": variation["sensitivity_weight"], "Alternative": names.get(row["alternative_id"], row["alternative_id"]), "Score": row["score"]})
                interval = (saved.get("comparison") or {}).get("winner_change_interval")
                if interval:
                    intervals.append({"Criterion": variation["sensitivity_criterion"], "Tested interval": str(interval)})
            except (OSError, ValueError, TypeError, KeyError) as error:
                st.warning(f"Cannot load sensitivity case {item['case_key']}: {error}")
        if points:
            st.subheader("How stable is the leading option?")
            frame = pd.DataFrame(points)
            criterion = st.selectbox("Sensitivity criterion", sorted(frame["Criterion"].unique()))
            filtered = frame[frame["Criterion"] == criterion]
            st.altair_chart(alt.Chart(filtered).mark_line(point=True).encode(x=alt.X("Weight:Q", axis=alt.Axis(format=".0%")), y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])), color="Alternative:N", tooltip=["Alternative", "Weight", alt.Tooltip("Score:Q", format=".4f")]), width="stretch", alt="Saved TOPSIS scores at tested criterion weights")
            st.caption("Points are saved evaluations; connecting lines are visual guides. Winner changes are bounded by tested intervals, not exact thresholds.")
            table([i for i in intervals if i["Criterion"] == criterion], "Saved winner-change intervals")

elif view == "Calculation trace":
    st.subheader("Inspect the scientific trace")
    st.caption(f"Preference source: {preferences.get('source', 'unavailable')}. Original preference weights and applied TOPSIS weights may differ.")
    if preferences:
        levels = {"Session": [session], "Groups": preferences["groups"], "Participants": preferences["participants"]}
        level = st.selectbox("Aggregation level", list(levels))
        if levels[level]:
            selected_id = st.selectbox("Judgment trace", [i["id"] for i in levels[level]])
            item = next(i for i in levels[level] if i["id"] == selected_id)
            matrix(item["matrix"], "Pairwise comparison matrix")
            table([{"Criterion": k, "Original preference weight": v} for k, v in zip(preferences["criterion_ids"], item["weights"])], "Original preference weights")
            st.json({k: v for k, v in item.items() if k not in {"matrix", "weights"}})
    if result["topsis"]:
        stage = st.selectbox("TOPSIS matrix", ["decision_matrix", "normalized_matrix", "weighted_matrix"], format_func=lambda s: s.replace("_", " ").capitalize())
        matrix(result["topsis"][stage], stage.replace("_", " "))
        with st.expander("Ideals, distances and full TOPSIS trace"):
            st.json(result["topsis"])
    else:
        st.info("No TOPSIS trace is available for this run.")

elif view == "Export":
    st.subheader("Take the evidence with you")
    st.caption(f"Run ID: {result['run_id']} · Schema: {result['schema_version']} · Batch: {batch.name}")
    with st.container(horizontal=True):
        st.download_button("Selected run JSON", payload, file_name=f"{result['run_id']}.json", mime="application/json", icon=":material/download:")
        if rankings:
            st.download_button("Selected ranking CSV", pd.DataFrame(rankings).to_csv(index=False), file_name=f"{case}_ranking.csv", mime="text/csv")
    st.markdown("**Full batch exports**")
    st.caption("These files contain all runs in the selected batch, regardless of the scenario filter.")
    with st.container(horizontal=True):
        for filename in ["summary.csv", "rankings.csv", "weights.csv", "assertions.csv"]:
            path = batch / filename
            if path.is_file() and path.resolve().parent == batch.resolve():
                st.download_button(filename, path.read_bytes(), file_name=filename, mime="text/csv")
    with st.expander("Reproducibility and provenance", expanded=True):
        st.json(result["provenance"])

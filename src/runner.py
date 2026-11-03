"""Resolve declarative variations, execute dependencies, and persist scientific runs."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np

from src.assertions import evaluate_assertions, validate_assertions
from src.experiment import Experiment
from src.reference import DERIVATIONS, compare, evaluate, validate_ratio
from src.results import BatchStore, RunResult, now
from src.variations import ExecutionVariation

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Case:
    key: str
    experiment: Experiment
    variation: ExecutionVariation | None = None

    @property
    def assertions(self):
        base = self.experiment.expected_assertions
        extra = self.variation.expected_assertions if self.variation else []
        return [*base, *extra,
                {"id": "weights_normalized", "check": "normalized_weights", "expected": True},
                {"id": "scores_bounded", "check": "bounded_topsis_scores", "expected": True}]

    @property
    def parent(self):
        return self.variation.parent if self.variation else None

    @property
    def comparison_key(self):
        return (self.variation.compare_to or self.parent) if self.variation else None

    @property
    def dependencies(self):
        return list(dict.fromkeys(k for k in (self.parent, self.comparison_key) if k))


def normalized_inputs(experiment):
    session = experiment.session
    return {"experiment_schema_version": experiment.schema_version,
            "research_purpose": experiment.research_purpose,
            "scenario": {"id": experiment.scenario_id, "description": experiment.scenario_description,
                         "policy_question": experiment.scenario_policy_question},
            "criteria": [c.model_dump(mode="json") for c in session.criteria],
            "preference_criteria": [c.model_dump(mode="json") for c in session.criteria],
            "questions": [q.model_dump(mode="json") for q in session.questions],
            "groups": [g.model_dump(mode="json") for g in session.stakeholder_groups],
            "participants": [{"id": p.id, "group_id": p.stakeholder_group_id,
                              "judgments": {r.question_id: float(r.preference.ahp_scale if hasattr(r.preference, "ahp_scale") else r.preference) for r in p.responses}}
                             for p in session.participants],
            "alternatives": [{"id": a.id, "name": a.name, "description": a.description,
                              "values": {v.criterion_id: v.value for v in a.criterion_values}} for a in session.alternatives],
            "processing": {"weighting_method": experiment.processing.weighting_method.method.value,
                           "ranking_method": experiment.processing.ranking_method.method.value,
                           "wd": experiment.processing.weighting_method.parameters.get("wd", "geometric")},
            "consistency_threshold": experiment.execution.consistency_threshold,
            "random_seed": experiment.execution.random_seed, "weights": None}


def _known(keys, allowed, label):
    unknown = set(keys) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {label}: {sorted(unknown)}")


def validate_case(case):
    e, v = case.experiment, case.variation
    validate_assertions(case.assertions)
    p = e.processing
    if p.weighting_method.method.value != "ahp" or p.ranking_method.method.value != "topsis":
        raise ValueError("The reference runner supports AHP and TOPSIS only.")
    if set(p.weighting_method.parameters) - {"wd"} or p.ranking_method.parameters:
        raise ValueError("Unsupported calculation parameters.")
    if p.weighting_method.parameters.get("wd", "geometric") not in DERIVATIONS:
        raise ValueError("Unsupported AHP wd.")
    x = normalized_inputs(e)
    cids, gids, pids, qids = ([c["id"] for c in x["criteria"]], [g["id"] for g in x["groups"]],
                              [p["id"] for p in x["participants"]], [q["id"] for q in x["questions"]])
    if not 2 <= len(cids) <= 15:
        raise ValueError("AHP supports 2–15 criteria.")
    if not x["alternatives"]:
        raise ValueError("At least one alternative is required.")
    if not v:
        return
    counts = {g: sum(p["group_id"] == g for p in x["participants"]) for g in gids}
    counts.update(v.participant_counts)
    powers = {g["id"]: v.voting_powers.get(g["id"], g["voting_power"]) for g in x["groups"]}
    if not any(counts[g] > 0 and powers[g] > 0 for g in gids):
        raise ValueError("At least one represented group needs positive voting power.")
    _known(v.group_judgments, gids, "groups")
    _known(v.participant_judgments, pids, "participants")
    _known(v.participant_counts, gids, "groups")
    _known(v.voting_powers, gids, "groups")
    _known(v.target_profiles, gids, "target profile groups")
    _known(v.remove_criteria, cids, "criteria")
    for judgments in [*v.group_judgments.values(), *v.participant_judgments.values()]:
        if set(judgments) != set(qids):
            raise ValueError("Variation judgments must cover every question exactly once.")
        for ratio in judgments.values():
            validate_ratio(ratio)
    for gid, count in v.participant_counts.items():
        if count < 0 or isinstance(count, bool):
            raise ValueError("Participant counts must be nonnegative integers.")
        if count and not any(p["group_id"] == gid for p in x["participants"]):
            raise ValueError("Cannot replicate a group without base participants.")
    if any(not np.isfinite(n) or n < 0 for n in v.voting_powers.values()):
        raise ValueError("Voting powers must be finite and nonnegative.")
    if len(set(v.remove_criteria)) != len(v.remove_criteria) or len(v.remove_criteria) >= len(cids):
        raise ValueError("Criterion removal must leave at least one criterion, without duplicates.")
    retained = set(cids) - set(v.remove_criteria)
    if v.weights is not None:
        if set(v.weights) != retained or any(not np.isfinite(w) or w < 0 for w in v.weights.values()) or sum(v.weights.values()) <= 0:
            raise ValueError("Explicit weights must cover retained criteria and have a positive finite total.")
    if (v.reuse_parent_weights or v.remove_winner or v.inherit_parent_alternatives) and not v.parent:
        raise ValueError("Parent-dependent variation requires parent.")
    if v.remove_criteria and not v.reuse_parent_weights and v.weights is None:
        raise ValueError("Criterion removal requires reused or explicit weights.")
    if v.reuse_parent_weights and v.weights is not None:
        raise ValueError("Choose explicit weights or reused weights.")
    for alternative in v.add_alternatives:
        if set(alternative.values) != set(cids) or not all(np.isfinite(n) for n in alternative.values.values()):
            raise ValueError("Added alternatives need finite values for every criterion.")
    for profile in v.target_profiles.values():
        if len(profile) != len(cids) or any(not np.isfinite(w) or w <= 0 for w in profile) or not np.isclose(sum(profile), 1):
            raise ValueError("Target profiles must be positive normalized criterion vectors.")
    if v.sensitivity_criterion is not None and v.sensitivity_criterion not in retained:
        raise ValueError("Unknown sensitivity criterion.")


def build_plan(experiments, scenario=None):
    cases = {}
    for e in experiments:
        variants = e.execution.variations
        expanded = [Case(v.variant_id, e, v) for v in variants] if variants else [Case(e.variant_id or e.id, e)]
        for c in expanded:
            if c.key in cases:
                raise ValueError(f"Duplicate run key: {c.key}")
            validate_case(c)
            cases[c.key] = c
    def normalize(value):
        match = re.fullmatch(r"S0*(\d+)", value, re.I)
        return f"S{int(match[1]):02}" if match else value
    targets = [c.key for c in cases.values() if scenario is None or normalize(c.experiment.id) == normalize(scenario) or c.key == scenario]
    if not targets:
        raise ValueError(f"No matching scenario: {scenario}")
    ordered, visiting, visited = [], set(), set()
    def visit(key):
        if key not in cases:
            raise ValueError(f"Missing parent/comparison run: {key}")
        if key in visiting:
            raise ValueError(f"Dependency cycle at {key}")
        if key in visited:
            return
        visiting.add(key)
        for dep in cases[key].dependencies:
            visit(dep)
        visiting.remove(key)
        visited.add(key)
        ordered.append(cases[key])
    # Validate the entire graph even when selecting a subset.
    for key in cases:
        visit(key)
    alternative_ids = {}
    for case in ordered:
        v = case.variation
        ids = {a.id for a in case.experiment.session.alternatives}
        if v:
            if v.inherit_parent_alternatives:
                ids = set(alternative_ids[v.parent])
            if len(set(v.remove_alternatives)) != len(v.remove_alternatives):
                raise ValueError("Alternative removals contain duplicates.")
            _known(v.remove_alternatives, ids, "alternatives to remove")
            ids -= set(v.remove_alternatives)
            for a in v.add_alternatives:
                if a.id in ids:
                    raise ValueError(f"Duplicate added alternative: {a.id}")
                ids.add(a.id)
            if not ids or (v.remove_winner and len(ids) < 2):
                raise ValueError("Alternative removal must leave at least one alternative.")
        alternative_ids[case.key] = ids
    selected = set()
    def include(key):
        if key not in selected:
            selected.add(key)
            for dep in cases[key].dependencies:
                include(dep)
    for key in targets:
        include(key)
    plan = [c for c in ordered if c.key in selected]
    # Cross-variant assertion needs both actual S02 results.
    if any(c.experiment.id == "S02" for c in plan):
        for c in ordered:
            if c.experiment.id == "S02" and c.key not in selected:
                plan.append(c)
    return plan


def resolve_inputs(case, completed):
    x = normalized_inputs(case.experiment)
    v = case.variation
    if not v:
        return x, None
    parent = completed[case.parent] if case.parent else None
    if v.inherit_parent_alternatives:
        x["alternatives"] = deepcopy(parent.inputs["alternatives"])
    for g in x["groups"]:
        g["voting_power"] = v.voting_powers.get(g["id"], g["voting_power"])
    for p in x["participants"]:
        if p["group_id"] in v.group_judgments:
            p["judgments"] = deepcopy(v.group_judgments[p["group_id"]])
        if p["id"] in v.participant_judgments:
            p["judgments"] = deepcopy(v.participant_judgments[p["id"]])
    for gid, count in v.participant_counts.items():
        templates = [p for p in x["participants"] if p["group_id"] == gid]
        x["participants"] = [p for p in x["participants"] if p["group_id"] != gid]
        for i in range(count):
            p = deepcopy(templates[i % len(templates)])
            p["id"] = f"{gid}_p{i+1:02}"
            x["participants"].append(p)
    removals = list(v.remove_alternatives)
    if v.remove_winner:
        removals.append(parent.topsis["rankings"][0]["alternative_id"])
    _known(removals, [a["id"] for a in x["alternatives"]], "alternatives to remove")
    x["alternatives"] = [a for a in x["alternatives"] if a["id"] not in removals]
    for a in v.add_alternatives:
        if a.id in {old["id"] for old in x["alternatives"]}:
            raise ValueError(f"Duplicate alternative: {a.id}")
        x["alternatives"].append(a.model_dump())
    x["criteria"] = [c for c in x["criteria"] if c["id"] not in v.remove_criteria]
    x["variation"] = v.model_dump()
    x["target_profiles"] = v.target_profiles
    reused = None
    if v.reuse_parent_weights or v.weights is not None:
        # Preference judgments remain unchanged for the sensitivity experiments.
        if parent is not None:
            if any(x[key] != parent.inputs[key] for key in ("participants", "groups", "preference_criteria", "questions", "processing", "consistency_threshold")):
                raise ValueError("Cannot reuse parent preference trace after changing preference inputs or method settings.")
            reused = deepcopy(parent.preferences)
            reused["source"] = f"reused:{parent.run_id}"
        x["weights"] = deepcopy(v.weights if v.weights is not None else {c["id"]: parent.applied_weights[c["id"]] for c in x["criteria"]})
    return x, reused


def _hash(payload):
    return hashlib.sha256(payload).hexdigest()


def provenance(experiment, inputs):
    hashes = {}
    source = Path(experiment.source_path)
    for label, path in (("experiment", source), ("criteria", source.parent / experiment.resources.criteria),
                        ("alternatives", source.parent / experiment.resources.alternatives), ("dataset", source.parent / experiment.resources.dataset)):
        hashes[label] = _hash(path.read_bytes())
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    versions = {}
    for package in ("numpy", "scipy", "pydantic", "pandas", "streamlit"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    # Include untracked source files: git revision alone cannot identify this workspace.
    code = {str(p.relative_to(ROOT)): _hash(p.read_bytes()) for p in sorted((ROOT / "src").rglob("*.py"))}
    for name in ("run_experiment.py", "app.py"):
        code[name] = _hash((ROOT / name).read_bytes())
    return {"resource_hashes": hashes, "input_hash": _hash(json.dumps(inputs, sort_keys=True, allow_nan=False).encode()),
            "code_revision": git("rev-parse", "HEAD"), "dirty_worktree": bool(git("status", "--porcelain")),
            "code_hashes": code, "python": sys.version, "dependencies": versions, "seed": experiment.execution.random_seed,
            "engine": "independent_numpy_reference", "poliinsight_validated": False}


def execute_batch(plan, output):
    store = BatchStore(output)
    completed = {}
    for case in plan:
        start, tick = now(), time.perf_counter()
        e = case.experiment
        result = RunResult(experiment_id=e.id, variant_id=case.variation.variant_id if case.variation else e.variant_id,
                           case_key=case.key, name=e.name, run_id=uuid4().hex,
                           parent_run_id=completed[case.parent].run_id if case.parent else None, status="failed")
        result.execution = {"started_at": start, "messages": [], "description": case.variation.description if case.variation else e.variant_description}
        try:
            result.provenance = provenance(e, normalized_inputs(e))
            if any(completed[d].status != "passed" for d in case.dependencies):
                result.status = "blocked"
                result.execution["messages"].append("A required parent or comparison run failed.")
            else:
                result.inputs, reused = resolve_inputs(case, completed)
                result.provenance = provenance(e, result.inputs)
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    trace = evaluate(result.inputs, reused)
                result.preferences, result.applied_weights, result.topsis = trace["preferences"], trace["applied_weights"], trace["topsis"]
                result.status = "passed"
                if case.comparison_key:
                    other = completed[case.comparison_key]
                    result.comparison = {"reference_run_id": other.run_id, "reference_case_key": other.case_key, **compare(trace, other.model_dump())}
                    v = case.variation
                    previous = other.inputs.get("variation", {})
                    if v and v.sensitivity_criterion and previous.get("sensitivity_criterion") == v.sensitivity_criterion and result.comparison["winner_changed"]:
                        result.comparison["winner_change_interval"] = [previous["sensitivity_weight"], v.sensitivity_weight]
                result.assertions = evaluate_assertions(case.assertions, result.model_dump(), completed, pending_sibling=True)
                if any(a["status"] in {"failed", "error"} for a in result.assertions):
                    result.status = "failed"
        except Exception as error:
            result.status = "failed"
            result.execution["messages"].append(f"{type(error).__name__}: {error}")
        result.execution.update(completed_at=now(), duration_seconds=time.perf_counter() - tick)
        completed[case.key] = result
    for case in plan:
        r = completed[case.key]
        if any(a["status"] == "pending" for a in r.assertions):
            r.assertions = evaluate_assertions(case.assertions, r.model_dump(), completed)
            if any(a["status"] in {"failed", "error"} for a in r.assertions):
                r.status = "failed"
    for case in plan:
        r = completed[case.key]
        if any(completed[d].status != "passed" for d in case.dependencies) and r.status == "passed":
            r.status = "blocked"
            r.execution["messages"].append("A dependency failed a batch-level assertion.")
        store.save(r)
    store.finalize(list(completed.values()))
    return store.path, list(completed.values())

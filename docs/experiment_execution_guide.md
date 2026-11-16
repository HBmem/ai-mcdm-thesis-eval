# Building `execute_experiments`: first numerical increment

This increment implements the three requested capabilities. Method adapters live
in `src/mcdm/execution.py`; batch orchestration lives in
`src/experiments/runner.py`. They consume your existing `PreparedRun` and
`Experiment` objects. The executor does not reload JSON or regenerate participants.

```python
from src.experiments.loader import load_experiments
from src.experiments.runner import prepare_runs, execute_experiments

runs = prepare_runs(load_experiments())
artifact_path, results = execute_experiments(runs)

assert artifact_path is None  # Saving is a later increment.
for result in results:
    print(result["execution_key"], result["status"])
    if result["status"] == "calculated":
        print(result["trace"]["applied_weights"])
        print(result["trace"]["topsis"]["rankings"])
```

Numerical success is `calculated`, not `passed`. Input assertions have been
validated by preparation, but they have not been evaluated against outputs.

## Step 1: Select methods from `experiment.processing`

JSON settings already reach these objects through your loader:

```python
process = experiment.processing
weighting = process.weighting_method
ranking = process.ranking_method
aggregation = process.aggregation_method
```

The execution module provides four registries:

```python
WEIGHTING_METHODS[weighting.method]
RANKING_METHODS[ranking.method]
WITHIN_GROUP_METHODS[aggregation.within_group]
BETWEEN_GROUP_METHODS[aggregation.between_groups]
```

A weighting or ranking entry is a `MethodAdapter(validate, calculate)`. Preparation
and execution call the same `validate_methods()` capability check. New adapters
therefore do not require a second hard-coded method list in `prepare_runs()`.

The implemented methods are:

| Stage | Method | Parameters |
| --- | --- | --- |
| Weighting | AHP through PyDecision | `derivation`: mean, geometric, max_eigen, or supported aliases |
| Ranking | TOPSIS through PyDecision | `normalization`: vector |
| Within group | Geometric mean of judgment matrices | Equal participant influence within each group |
| Between groups | Weighted geometric mean of group judgment matrices | Voting powers normalized across represented groups |

Unknown method names and unsupported parameters fail explicitly. The adapter maps
JSON's `derivation` parameter to PyDecision's `wd` argument. This avoids the current
unused `_normalize_inputs()` helper's mistake of looking for `wd` in the JSON
parameters and silently falling back to geometric weighting.

**Next:** derive weights at every entity level using the selected adapter.

**Unfinished/missing:** only these methods are implemented. Add an enum value,
parameter validator, calculation adapter, registry entry, and method-specific
analytical tests for a new method. The current domain still describes pairwise
judgments; methods requiring other input forms need corresponding models.

**Difference from the old version:** the independent reference executor calculated
weights itself. This path delegates AHP and TOPSIS to PyDecision. It identifies its
engine as PyDecision and records the installed version. Group aggregation remains
explicit local matrix logic; PyDecision's AHP/TOPSIS functions are not the group
aggregation policy.

**Robustness:** keep the reference implementation as a separate comparison oracle.
Do not describe these new runs as independent validation of PyDecision. Never
silently ignore a parameter or substitute another algorithm.

## Step 2: Generate individual, group, and session weights

The calculation order is:

1. Build one reciprocal criterion-comparison matrix per participant using the
   question definitions and `response.preference.ahp_scale`.
2. Run the configured weighting adapter for every participant matrix.
3. Aggregate participant judgment matrices within each represented group, then run
   the same weighting adapter on each group matrix.
4. Aggregate group matrices using normalized group voting powers, then run the
   same weighting adapter on the session matrix.

This is aggregation of individual judgments (AIJ): aggregate matrices before
weight derivation. It does not average individual weights into group weights or
average participant rankings into a session ranking.

The central call is:

```python
handler = WEIGHTING_METHODS[process.weighting_method.method].calculate
result = handler(
    matrix,
    process.weighting_method.parameters,
    process.consistency_method.threshold,
)
```

The PyDecision adapter calls:

```python
weights, cr = ahp_method(matrix.copy(), wd=derivation)
```

It validates the returned weights and consistency ratio. For two criteria, it
keeps PyDecision's weights and records the CR=0 convention explicitly: the library's
RI table has a zero denominator at this size. Inconsistent judgments are retained
and flagged using the configured threshold.

Empty groups are excluded under the current policy and their IDs are recorded.
Their voting powers do not dilute the represented groups' influence. With five
represented groups of power 0.2, increasing one group from one to 30 copies of an
identical participant leaves its group voting power at 0.2.

Results are under:

```python
trace["preferences"]["participants"]
trace["preferences"]["groups"]
trace["preferences"]["session"]
trace["applied_weights"]  # Criterion-ID map for session weights.
```

Each preference result includes an ordered labeled matrix, weights, consistency
ratio, threshold, flag, and derivation. Group results include participant count,
assigned voting power, and effective normalized voting power.

**Next:** rank the same alternatives with each entity's own weight vector.

**Unfinished/missing:** participant-level voting powers, alternative aggregation
policies, incomplete responses, and fuzzy judgments are not implemented. The
current planner requires complete pairwise responses and 2–15 criteria.

**Difference from the old version:** group/session aggregation semantics are
preserved. Every participant and group now receives a direct PyDecision weight
calculation through the same configured path as the session. The older legacy
service called PyDecision AHP primarily for the aggregated session.

**Robustness:** preserve the ordering of criterion IDs, retain inconsistency
warnings at every level, and compare all derivations against analytically
consistent matrices. A consistent session matrix does not establish that each
individual participant was consistent.

## Step 3: Rank alternatives for each entity

The executor builds one alternatives-by-criteria decision matrix. Values are
aligned by criterion ID, so the order of `criterion_values` within an alternative
cannot change the result. Each ranking uses that entity's criterion weights:

```python
scores = topsis_method(
    decision_matrix.copy(),
    entity_weights.copy(),
    criterion_directions,  # "min" or "max", in criterion order.
    graph=False,
    verbose=False,
)
```

PyDecision returns scores in input-row order. The adapter associates each score
with the corresponding alternative ID before sorting. It stores ordinal display
ranks, average statistical ranks for score ties within 1e-12, and the complete
winner set. Ordinal tie ordering follows the decision matrix's input row order.

The result structure is:

```python
trace["participant_rankings"][participant_id]
trace["group_rankings"][group_id]
trace["topsis"]  # Session ranking.
```

Every ranking includes its own applied weights and labeled decision, normalized,
and weighted matrices. These intermediate matrices and ideal points are
reconstructed locally from the library inputs for audit; the scores themselves
come from PyDecision.

**Next:** persist a versioned result, then evaluate local assertions and
cross-run comparisons in separate stages.

**Unfinished/missing:** this increment deliberately rejects all-zero criterion
columns and cases where all effectively weighted alternatives coincide, including
a single remaining alternative. These cases make the installed PyDecision TOPSIS
formula undefined. It does not silently replace undefined scores.

**Difference from the old version:** the independent engine handled zero columns
and assigned score 0.5 to identical alternatives. The older legacy service also
implemented TOPSIS itself. This path calls PyDecision TOPSIS directly and adds
participant/group rankings to the execution trace.

**Robustness:** choose and version an explicit boundary policy next. If you restore
neutral scores for degenerate cases, label that as an adapter convention rather
than an output produced by PyDecision. Record ties as winner sets rather than
arbitrarily treating the first row as the sole winner.

## What `execute_experiments()` now does

It assigns a fresh attempt UUID, retains prepared keys, seeds, repetitions,
reference mappings, and input snapshots, calls the numerical engine, and returns
one in-memory result dictionary per run. Failed calculations do not prevent
independent runs from executing. A failed hard numerical dependency blocks its
consumer; a comparison reference alone does not block calculation.

Only the numerical step can become `calculated` in this increment. Assertions,
comparisons, and storage remain `not_run`. AI requests raise `NotImplementedError`
instead of appearing successful without an AI call. Supplying only one of an AI
configuration or client raises `ValueError`.

The `output` argument is reserved and does not write anything. The return value
is `(None, results)` to preserve the CLI's two-value unpacking while making the
absence of an artifact path explicit. These dictionaries are not yet historical
`RunResult` artifacts, and must not be passed directly to the old dashboard.

## Next implementation steps

1. Repair and version the result contract. `src/experiments/results.py` currently
   declares defaulted dataclass fields before required fields. `@dataclass(kw_only=True)`
   is one way to fix construction, but also update its post-init contracts for
   `calculated` versus final `passed` and the new participant/group rankings. Its
   commented-out comprehension field is still referenced by schema-1.1 validation.
   The runner no longer imports this unfinished model merely to prepare runs.
2. Add an artifact writer with atomic saves, a manifest, and stage-level
   checkpoints. A ranking error currently discards the in-memory weight trace for
   that failed calculation; checkpointing should preserve successful earlier
   stages. Saving is the next concrete implementation target.
3. Evaluate numerical assertions against the trace. Change a run to `passed` only
   after all required stages/checks succeed. Warning-severity checks should remain
   distinguishable from errors.
4. After numerical executions complete, use each `PreparedRun.reference_runs`
   mapping to calculate comparisons and comparison assertions. Avoid tying their
   order to JSON file order. Reference pairing is already handled by preparation.
5. Correct and activate provenance. The existing unused helper has the wrong root
   (`src` instead of the repository), its nested `git()` function does not return
   stdout, and its engine label still says independent NumPy. Record PyDecision
   version, full processing configuration, actual seed, input/resource/code hashes,
   and the package-to-output attribution already present in the numerical trace.
6. Update CLI reporting and exit codes for calculated/failed/blocked runs. Current
   `main.py` unpacks results without reporting numerical failure. For AI later,
   move executor calls inside `with ai_config.client() as client:`: the current
   AI branch closes that client before handing it to the executor.
7. Implement result-dependent interventions explicitly. Hard dependency ordering
   exists, but operations such as removing a computed winner or reusing weights
   need artifact-consumption rules, not just a dependency edge.

## Checks and sources

```bash
MPLCONFIGDIR=/tmp/ai-mcdm-matplotlib .venv/bin/python -m unittest discover -s tests -p test_execution.py -v
MPLCONFIGDIR=/tmp/ai-mcdm-matplotlib .venv/bin/python -m unittest discover -s tests -p test_prepare_runs.py -v
```

PyDecision is now a runtime dependency in `requirements.txt`. The adapters were
checked against the installed package source as well as the upstream
[AHP implementation](https://github.com/Valdecy/pyDecision/blob/master/pyDecision/algorithm/ahp.py)
and [TOPSIS implementation](https://github.com/Valdecy/pyDecision/blob/master/pyDecision/algorithm/topsis.py).

# Preparing executions from loaded cases

`src/experiments/runner.py` now provides `prepare_runs()`. It accepts the objects
returned by `load_experiments()` and produces a list of `PreparedRun` objects.
The loader creates cases; the planner selects and schedules their executions.

## Usage

```python
from src.experiments.loader import load_experiments
from src.experiments.runner import prepare_runs

experiments = load_experiments()
runs = prepare_runs(experiments, scenario="S05/unequal_participation")

for run in runs:
    print(run.key, run.repetition, run.seed, run.requested)
```

With the current JSON files this prints:

```text
S01/control#r001 1 123456 False
S05/control#r001 1 123456 False
S05/unequal_participation#r001 1 123456 True
```

The requested case compares with `S05/control`. That control compares with
`S01/control`, so the planner includes both references. `requested=False`
distinguishes supporting cases from cases selected directly by the caller.

Other selectors:

```python
all_runs = prepare_runs(experiments)
s05_runs = prepare_runs(experiments, scenario="S05")
same_runs = prepare_runs(experiments, scenario="S5")
selected = prepare_runs(experiments, scenario=["S01", "S05/control"])
```

Each selector must match. An empty selector list or an unmatched selector raises
`ValueError`. Load the full experiment collection before selecting cases so their
references are available. Passing only S05 cases is insufficient while its control
references S01. The planner does not read files to discover missing cases.

## PreparedRun fields

| Field | Meaning |
| --- | --- |
| `key` | Deterministic execution key, such as `S05/control#r002` |
| `case_key` | Unique loaded case identity, such as `S05/control` |
| `repetition` | One-based execution number within that case |
| `seed` | Effective seed for this execution |
| `experiment` | Independent deep copy of the resolved case inputs and metadata |
| `requested` | Whether the caller selected this case directly |
| `dependencies` | Execution keys whose calculation results must be available first |
| `reference_runs` | Mapping from referenced case keys to specific execution keys |

`key` is unique within this plan. It is intentionally stable across calls for
inspection and linking; the executor must assign a fresh persistent `run_id`
when saving an actual execution or retry. Avoid keying runs by `experiment.id`,
because that family identifier is shared by all S05 cases.

## Section 1: register and validate resolved cases

The first block in `prepare_runs()` builds three dictionaries:

```python
catalog     # case key -> loaded Experiment
references  # case key -> baseline/comparison/assertion reference keys
dependencies  # case key -> explicit result dependency keys
```

`_case_key()` verifies that the variant and session identifiers agree and match
`experiment_id/case_id`. Duplicate case keys are rejected.

`_validate_case()` checks schema version, pipeline, methods, supported parameters,
aggregation, consistency settings, repetitions, seeds, numerical input shape,
complete participant responses, and retained `PairwisePreference` enum values.
Legacy unexpanded `execution.variations` are rejected instead of expanded twice.

`validate_assertions()` in `src/experiments/assertions.py` checks the named assertion
contracts used by S01/S05: unique IDs, allowed parameter sets, expected entity IDs,
finite expectations, valid tolerance values, and calculation/comparison stages.
It returns assertion reference keys to include in the plan. The registry describes
declarations; result evaluators still need implementation. No assertion passes or
fails scientifically during preparation.

`_references()` adds baseline and explicit comparison references, and validates
comparison IDs, metrics, alignment policies, and tie tolerance. A baseline is made
available for reporting; this does not manufacture an extra comparison assertion.

All supplied cases are validated before selection, so filtering does not hide
an invalid case elsewhere in the supplied experiment collection.

## Section 2: validate links and order result dependencies

Every reference must exist in `catalog`. Explicit comparisons currently require
identical criterion and alternative ID sets, matching their declared
`require_identical_ids` policies. Supporting removal/addition comparisons will
require an additional alignment policy and a corresponding evaluator.

The recursive `order_case()` visits hard dependencies first, then the consuming
case. The `visiting` set detects dependency cycles; `visited` avoids duplicate work.
Sorted traversal makes the resulting order independent of input-file iteration.

Only explicit `case_metadata["result_dependencies"]` defines a hard execution edge:

```json
"result_dependencies": ["S01/control"]
```

This is a scheduling declaration copied from the case JSON by the loader. It does
not itself implement weight reuse, winner removal, or any result-consuming
operation. The future executor must implement and honor those operations.

The three relationships have separate roles:

- `input_from`: already materialized by the loader; no execution edge is added.
- Baseline/comparison/assertion references: results are needed for later reporting
  or evaluation, but they do not determine calculation order.
- `result_dependencies`: another calculation must finish before this one starts.

Reciprocal comparisons are valid. Reciprocal hard dependencies are cycles and fail.
The executor should block a dependent calculation if its required calculation
fails; a failed comparison alone need not invalidate an otherwise usable calculation.

## Section 3: select cases

Family selectors such as `S5`, `S05`, and `s005` normalize to the same value.
Exact case selectors can also use an alias in their family prefix, such as
`S5/control`. Case-name suffixes remain case-sensitive.

`requested` contains the matched case keys. No selector means all cases.
An iterable of selectors selects their union.

## Section 4: include supporting references

The work list starts with the requested cases. For each newly included case, its
references and result dependencies are added to the work list. This continues
until the complete transitive set has been collected.

The `included` set prevents duplicate runs and stops reciprocal comparisons from
causing an infinite loop. A shared baseline is included once, not once per consumer.

## Section 5: pair repeated executions

`reference_run()` chooses the exact execution key for each reference:

- Equal repetition counts: repetition 2 refers to repetition 2.
- One reference repetition: every consumer repetition refers to reference run 1.
- Any other count combination: preparation raises `ValueError` because pairing
  would be ambiguous without an additional experiment policy.

These rules apply to reporting references and hard result dependencies. Averaging
reference repetitions or selecting an arbitrary first trial is never implicit.

## Section 6: construct independent executions

For every included case, the final loop creates `defaults.repetition` records
(the Python field currently uses singular `repetition`; JSON uses `repetitions`).
Each record gets a deep copy of the loaded case and its effective seed:

```python
seed = experiment.defaults.seed + repetition - 1
```

For base seed 123456, three repetitions receive 123456, 123457, and 123458.
The copy's `execution.random_seed` and `defaults.seed` are updated together.
The source objects remain unchanged, and mutations during one execution cannot
leak into another execution's input dictionaries or session models.

Participant generation is not repeated. Every trial retains the cohort created
by the loader. A changing seed therefore does not change deterministic MCDM
results or create a new sampled cohort. Explicit cohort resampling and AI trial
semantics require their own future configuration.

## Integration with the executor

The function returns a list ordered for calculation dependencies. The executor
should calculate those runs first, recording successes/failures, then evaluate
comparisons and cross-run assertions using `reference_runs`. Comparison cycles
remain safe because this happens after numerical calculation.

This planner performs no calculations, AI calls, participant generation, or file
writes. `execute_experiments()` remains an explicit unimplemented hook; it raises
`NotImplementedError` rather than reporting success. `prepare_experiments()` is
a compatibility wrapper for the runner skeleton's original preparation name.
Your in-progress CLI is unchanged; it can import and call `prepare_runs()` after
loading the experiments.

## Verification

```bash
.venv/bin/python -B -m unittest discover -s tests -p test_prepare_runs.py -v
```

Tests exercise selection and aliases, transitive references, independent repeats,
seed assignment, one-to-one/broadcast pairing, dependency ordering and cycles,
reciprocal comparisons, input-only inheritance, malformed assertions/settings,
and the absence of calculation or participant-generation calls during planning.

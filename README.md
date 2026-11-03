# Independent MCDM evaluation

A reproducible NumPy reference implementation of the AHP–TOPSIS experiments in
[evaluation.md](evaluation.md), with a Streamlit viewer for saved artifacts.
There are 40 executable variations across S1–S10.

The reference engine does **not** call `MCDMService` or `pyDecision`. Passing its
analytical checks and scenario invariants does not validate PoliInsight. External
application comparisons and AI interpretation evaluation are deferred.

## Setup and use

Use Python 3.12 or later, with a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiment.py validate --experiments experiments
python run_experiment.py run --experiments experiments --output results
streamlit run app.py -- --results results
```

Run a scenario or one variation, automatically including its dependencies:

```bash
python run_experiment.py run --scenario S08 --output results
python run_experiment.py run --scenario S09_REMOVE --output results
```

`S8` and `S08` are equivalent scenario selectors. Default experiment and output
locations are anchored to the repository; explicitly supplied CLI paths are
relative to the current directory. Resource paths **inside** an experiment are
relative to that experiment JSON file.

Exit codes: `0` for successful validation/execution, `1` for a calculation,
assertion, or artifact-write failure, and `2` for invalid configuration/arguments.
A failed parent blocks its dependents while independent cases continue.

The dashboard only reads saved results. Filters never run experiments. Its five
views show status/assertions, experimental variables, intermediate calculations,
baseline comparisons/sensitivity plots, and provenance/downloads.

## Input schemas and variations

Schema `1.0` preserves the existing codes `-4…4`: code `4` represents ratio `9`.
Schema `2.0` stores numeric ratios directly: value `4` represents ratio `4`.
The allowed ratios are `1…9` and their reciprocals; JSON reciprocal values are
numbers such as `0.3333333333333333`, not strings such as `"1/3"`.
All unique criterion pairs and every participant response must be present.

The experiment JSON determines the AHP derivation:

```json
{
  "processing": {
    "weighting_method": "ahp",
    "ranking_method": "topsis",
    "weighting": {"wd": "mean"},
    "ranking": {}
  }
}
```

Supported derivations: `mean`/`m`, `geometric`/`g`, and `max_eigen`/`me`.
The legacy service default (`geometric`) applies when `wd` is omitted. The bundled
experiments explicitly choose `mean`. Nested method objects with a `parameters`
field remain supported; conflicting duplicate parameter locations are rejected.
No fuzzy methods or extra TOPSIS parameters are implemented.

Schema 2 example fragment for a dependent criterion-removal variation:

```json
{
  "schema_version": "2.0",
  "execution": {
    "random_seed": 123456,
    "consistency_threshold": 0.10,
    "variations": [
      {
        "variant_id": "S07_C1",
        "description": "Remove implementation cost",
        "parent": "S01",
        "remove_criteria": ["C1"],
        "reuse_parent_weights": true
      }
    ]
  }
}
```

The complete executable examples are in `experiments/`. Typed variations support
`group_judgments`, `participant_judgments`, `participant_counts`, `voting_powers`,
`remove_alternatives`, `add_alternatives`, `remove_winner`, `remove_criteria`,
`weights`, `reuse_parent_weights`, and `inherit_parent_alternatives`.
`parent` establishes the dependency; `compare_to` can choose a different comparison
reference. Every variation begins with a fresh copy of its base experiment;
parent inputs are inherited only when an operation explicitly requests them.

Judgment replacements must include the complete question set. They are applied
before participant replication; replication cycles through the group's base
participants with deterministic IDs. Empty groups are excluded from aggregation.
Nonnegative effective voting powers are normalized over represented groups, with
at least one active group having positive power.

For S4, target weight ratios are quantized by logarithmic distance to the Saaty
scale. Equal-distance choices favor the intensity closest to one. The dashboard
shows target and achieved weights because quantization cannot generally reproduce
target profiles exactly. S5 uses these same heterogeneous profiles to isolate
voting power and participant-count changes. S10 uses two participants per group
so one inconsistent participant differs from one inconsistent group.

## Numerical and comparison conventions

- Aggregate judgments geometrically within groups and with normalized voting
  powers across groups. Derive weights only after aggregation.
- Evaluate participant, group, and session consistency using the configured AHP
  derivation. For mean/geometric methods, estimate lambda-max with
  `mean((A @ w) / w)`; for eigenvector derivation, use the principal eigenvalue.
  The RI table and convention are saved with the trace. Two-criterion CR is zero;
  preference calculations support 2–15 criteria.
- Flag `CR > consistency_threshold` (default `0.10`) and retain all judgments.
- TOPSIS uses vector column normalization and the declared `max`/`min` criterion
  directions. Zero-valued columns contribute zero; identical alternatives receive
  score `0.5`. Invalid/nonfinite data and nonpositive weight totals are rejected.
- S6 and S9 retain the parent weights; S7 restricts and renormalizes them. S8 uses
  explicit weights. These sensitivity cases reuse the original preference trace,
  whose original criterion IDs remain visible separately from applied weights.
- Presentation ranks are ordinal. Statistical ranks average tied positions within
  an absolute score tolerance of `1e-12`. Correlations rerank only common
  alternatives and use Spearman rho and Kendall tau-b. Undefined correlations are
  saved as JSON null with a reason.
- A pairwise reversal requires a strict order to reverse. Tie transitions and rank
  position changes are reported separately. Winner comparisons use the set of
  tied winners. S8 reports winner-change intervals between tested weights, never
  claims exact thresholds.

## Artifacts and extension points

Each invocation creates a new batch directory under the output root. Previous
batches are never reused. Files are written atomically:

```text
results/<timestamp>_<unique-id>/
  manifest.json       # batch status and run entries
  <run-uuid>.json     # versioned RunResult for each case
  summary.csv
  rankings.csv
  weights.csv
  assertions.csv
```

Run JSON stores fully resolved inputs, labeled matrices, group/session AHP
results, applied weights, TOPSIS stages, comparisons, assertions, timestamps,
resource/input hashes, source-code hashes, Git revision/dirty state, Python and
dependency versions. Values retain full numerical precision. Generated results
are gitignored; retain or archive the batch directories needed for your thesis.

The assertion registry is in `src/assertions.py`. Unknown checks fail validation.
Checks requiring PoliInsight results must explicitly set `"deferred": true` and
are displayed as deferred, never passed. Variation-specific assertions augment
base assertions. Every successful calculation also checks normalized weights and
bounded scores. Cross-variant S2 assertions are finalized after both variants run.

Main interfaces:

- `load_experiments(directory)` reads/validates resources and preserves legacy
  participant-variant expansion.
- `build_plan(experiments, scenario=None)` expands execution variations and orders
  dependencies, rejecting cycles and missing references.
- `evaluate(inputs, reused_preferences=None)` returns independent preference and
  TOPSIS traces with applied weights.
- `execute_batch(plan, output)` returns the new batch path and `RunResult` objects.

These boundaries allow a later PoliInsight export adapter without changing the
reference calculations or dashboard's saved-run model.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -q
```

Tests include analytical matrices, every Saaty intensity, all AHP derivations,
aggregation invariants, consistency retention, TOPSIS boundary cases, ID-aligned
comparisons, dependency failures, repeatability, CLI exit codes, and Streamlit
`AppTest` scenarios. The dev dependencies include `pyDecision` solely for the
existing legacy service regression tests.

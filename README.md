# Independent MCDM evaluation

A reproducible NumPy reference implementation of the AHP–TOPSIS experiments in
[evaluation.md](evaluation.md), with a Streamlit viewer for saved artifacts.
There are 40 executable variations across S1–S10.

The reference engine does **not** call `MCDMService` or `pyDecision`. Passing its
analytical checks and scenario invariants does not validate PoliInsight. External
application comparisons remain deferred. Each successful experiment also runs the
Data Comprehension layer against the saved numerical evidence.

## Setup and use

Use Python 3.12 or later, with a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiment.py validate --experiments experiments
cp .env.example .env
# Edit .env to set AI_PROVIDER, AI_MODEL, and the provider's API key.
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
assertion, comprehension, or artifact-write failure, and `2` for invalid configuration/arguments.
`validate` requires no AI configuration. `run` requires the selected provider, model,
and provider-specific API key, and rejects missing configuration before creating a
batch. Configuration loads from
the repository-root `.env` using python-dotenv, even when invoked from another
directory. Existing environment variables take precedence; a blank environment
variable is not replaced by the file. `.env` is gitignored and must stay private.
A failed parent blocks its dependents while independent cases continue.

The dashboard only reads saved results. Filters never run experiments. Its views show
status/assertions, experimental variables, intermediate calculations, baseline
comparisons/sensitivity plots, individual AI calls, and provenance/downloads.

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
  preflight.json      # successful CLI startup access check, when present
  <run-uuid>.json     # versioned RunResult for each case
  <run-uuid>.call.<call-uuid>.json  # atomically saved model attempt, including failures
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

## Data Comprehension layer

Every mathematically successful case runs three sequential calls: session
understanding, stakeholder-group profiles, then session profile. Deterministic
context assembly precedes those calls. Validated results are available as
`RunResult.comprehension.state`; `comprehension.context` supports fact retrieval.
Later explanation and advisory stages are outside this implementation.

The client is the OpenAI Python SDK configured for an OpenAI-compatible provider.
Set `AI_PROVIDER` to `openrouter`, `ollama`, or `openai`, set `AI_MODEL` to the
provider's exact model identifier, and store the corresponding
`OPENROUTER_API_KEY`, `OLLAMA_API_KEY`, or `OPENAI_API_KEY` outside source control.
Each provider has a default endpoint; `AI_BASE_URL` can override it. Optional
`AI_APP_URL` and `AI_APP_NAME` values are sent as OpenRouter attribution headers.
Neither credential values nor SDK request headers are saved. Tests inject clients;
production has no synthetic fallback.
`execute_batch(..., ai_config=..., ai_client=...)` is the injection boundary.

Prompts include the schema; responses must be strict JSON and pass Pydantic and
evidence validation. OpenRouter requests require a routed endpoint that supports
the structured-output parameters instead of silently accepting an incompatible
endpoint.
Each request has a 120-second timeout and up to two retries for transient transport
or rate/server errors. Every attempt has its own audit record. Malformed or
unsupported output fails immediately. Successful prior stages remain available for
audit, while subsequent stages stop. Required comprehension failures fail the case
and block dependent cases; existing deterministic calculation traces are retained.

Context uses resolved variation inputs, original preference dimensions, current
ranking dimensions, source hashes, processing versions, explicit entity references,
and deterministic evidence. Missing objectives, constraints, stakeholder statements,
and other absent metadata remain gaps. Model interpretations are never evidence for
other model interpretations: reused findings must retain original citations and
qualifications. Every structured factual claim is checked; prose is screened but
cannot be completely verified by these rules. Human review remains necessary.

The diagnostic suite reports within-group weight ranges and population standard
deviations; pairwise Spearman rho and Kendall tau-b for participant/group rankings;
leave-one-participant/group-out comparisons; criterion-weight factors of 0.8, 0.9,
1.1 and 1.2 with other weights proportionally rescaled; and single-criterion removal
with remaining weights normalized. Group/participant rankings use group/participant
preferences over current ranking criteria. Fixed applied experiment weights remain
fixed in omission experiments. Undefined correlations, empty-group omissions and
infeasible weight perturbations retain explicit reasons. Single-criterion omission
tests TOPSIS with retained weights, without deriving a replacement AHP model.
These diagnostics are descriptive and conditional, with no high/low agreement or
universal robustness classification. Participant influence cases are grouped without
participant identifiers; no population representativeness or causal effect is inferred.

Artifact schema `1.1` adds typed comprehension state and per-attempt call records.
The dashboard continues to read `1.0` artifacts. Call sidecars are saved immediately,
so they survive a later stage failure or interruption; the final run JSON embeds the
same records. Interrupted batches may require inspecting sidecars directly because
the run manifest is finalized separately. Historical batches remain immutable.
Fresh runs regenerate comprehension. `context_package` requires matching run,
source and configuration hashes before supplying a downstream task package.

The main dashboard's **AI calls** view associates the selected call and attempt
with its run, exact role-labelled messages, structured input, output schema, raw and
parsed response, verification results, receipt, and errors. It never invokes a model.
Prompts use aggregate allowlisted data; source metadata needing redaction is rejected
before sending. Export/display sanitization removes known credentials, recognizable
secret fields, bearer tokens and email addresses. Redacted records are labelled.
Arbitrary sensitive prose cannot be identified reliably: source metadata still needs
appropriate review. The separate `dashboard_demo` is unchanged.

Verification uses the test-only client in `tests/dcl_fakes.py`, including a localhost
OpenAI-compatible server for real SDK/CLI tests. These exercise transport integration
without measuring a live model's interpretation quality. A live Cloud experiment
requires configured credentials and an available model.

## Startup API-key and model verification

Before `run` creates a batch or calculates experiment results, it makes one short
Chat Completions request with the configured key, endpoint and model. The request
contains only a fixed connectivity-check message, uses `max_tokens=8`, has a
30-second timeout, and disables SDK retries. It consumes a small amount of Cloud
usage. Both normal and token-limited finishes can confirm model access; the check
does not require the model to echo an exact reply. This verifies current access,
not future availability, DCL schema adherence or interpretation quality.

An authentication/permission failure or rejected model/request returns exit code
`2`. Connection failures, timeouts, rate limits, provider errors and malformed
responses return `1`; these do not establish that the key or model is invalid.
Failed checks create no experiment artifacts. The successful probe is saved once
as `preflight.json` and is shown under **Batch startup check** in the dashboard's
**AI calls** view, separately from per-run comprehension calls. The CLI reuses and
closes the same SDK client for the experiment batch. `validate` remains offline.

`AIConfig.from_env(env_file=None)` disables file loading for isolated tests or
explicit environment-only use. Direct `execute_batch()` callers retain their
existing injection boundary and do not automatically perform a startup probe.

## Comprehension validation failures

Successful startup access does not guarantee a valid comprehension response. The
CLI now reports the failed phase and validation reasons (for example, missing
structured checks, invalid subject IDs or unqualified uncertainty), with full
per-claim details retained in the call audit. Dependent cases remain blocked when
a required parent fails.

Prompt version `dcl-prompt-v3` supplies stage-specific coverage rules, explicit
allowed subject IDs, valid finding examples and an evidence-populated output
template. The template is request guidance only: the application never merges it
into a response or silently repairs missing checks/citations. Unfilled instruction
placeholders and malformed or unsupported responses still fail validation.

Model context uses compact diagnostic summaries rather than repeating every full
comparison vector. All tested conditions, winner/rank/tie exceptions, correlation
outcomes and coverage gaps remain represented; full calculations remain in the
saved authoritative evidence for retrieval. Explicit robustness findings identify
which interventions changed the winner, including exceptions to apparent stability.
Code/dependency hashes remain audit metadata rather than repeated model evidence.
This reduces context-window exhaustion while preserving traceability and minority
outcomes. Generated prose is still screened rather than proven correct.

Agreement findings describe the modeled group results. Identical aggregate rankings
do not establish participant unanimity, population consensus or absence of minority
views. The session-profile template includes this qualification, a conditional
assessment, preference uncertainty and citations to participation/statement gaps.
These are supplied as request guidance; invalid model assertions are still rejected,
not rewritten or silently accepted. The conservative wording screen recognizes
explicit local negations (such as “does not establish participant unanimity”);
a negation in another sentence or contrastive clause does not exempt an assertion.

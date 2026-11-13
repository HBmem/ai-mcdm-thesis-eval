# Data Comprehension implementation handoff

The required flow is connected: deterministic context assembly → Session
Understanding → Stakeholder-Group Profiles → Session Profile. Experiments consume
typed validated state through `RunResult.comprehension.state`. Later explanation
and advisory layers remain outside this implementation.

## Gaps found and resolved

- There was no AI client or experiment-to-model execution path.
- `AIAnalysisContext` was a stub that failed to import because its field default
  was supplied twice; `prompt.py` was empty.
- Explanation contracts referenced an undefined `AnalysisMethod`, lacked a
  Session Profile, and did not carry machine-checkable claim values or lineage.
- The runner did not calculate group/participant ranking diagnostics, agreement,
  leave-one-out influence, or a per-run bounded robustness suite.
- Saved runs and the dashboard had no individual AI-call records or error view.
- Some scenario metadata is not recorded, and an alternative added by a variation
  lacks a description. These remain explicit gaps/null values rather than invented facts.

## Completed changes and files

| Files | Result |
| --- | --- |
| `src/models/analysis.py`, `src/models/explanation.py`, `src/models/prompt.py` | Versioned authoritative evidence, relationships, uncertainty, structured claims/checks, verification/review records, DCL output contracts, prompts and ordered stage inputs. |
| `src/comprehension.py` | Allowlisted context assembly, source/configuration hashes, fact retrieval and task packaging, strict JSON/schema/evidence validation, OpenAI SDK integration for Ollama Cloud, bounded retries and per-attempt audit records. |
| `src/diagnostics.py` | Participant/group TOPSIS rankings, weight ranges/population deviations, rank correlations, omission counterfactuals and bounded weight/criterion sensitivity. |
| `src/privacy.py` | Credential/recognizable sensitive-field redaction; metadata needing redaction is rejected before sending a prompt. |
| `src/runner.py`, `src/results.py`, `run_experiment.py` | Required comprehension, typed results, schema 1.1 with 1.0 reading support, atomic call sidecars, failure/dependency handling and configuration preflight. |
| `app.py` | Per-run call/attempt selection, exact saved messages, structured input/output, raw output, schemas, receipts, errors, verification and sanitized downloads. |
| `tests/dcl_fakes.py`, `tests/test_comprehension.py`, `tests/test_app.py`, `tests/test_runner.py` | Test-only model fixtures, real SDK/CLI transport against a localhost fixture, analytical diagnostics, failure handling, artifact and dashboard regression coverage. |
| `requirements.txt`, `README.md`, `DCL_IMPLEMENTATION.md` | SDK dependency, configuration/use documentation and this handoff. |

The existing whitespace change in `src/models/session.py` was preserved. The
separate `dashboard_demo` directory was left unchanged.

## Verified

- `.venv/bin/python -m unittest discover -s tests -q`: **62 tests passed** in
  134.232 seconds. This includes all 40 experiment variations with an injected
  client, analytical numerical fixtures, the actual OpenAI SDK against a local
  OpenAI-compatible HTTP server, CLI subprocess tests and Streamlit AppTest.
- `.venv/bin/python -m compileall -q src app.py run_experiment.py tests`: passed.
- `git diff --check`: passed.
- S01 `validate` works without AI credentials. S01 `run` without required
  environment variables returns configuration exit code 2 before creating output.
- Representative S01 with the deterministic test client: **passed**, with three
  successful stages, three saved call sidecars, five stakeholder descriptions in
  Session Understanding, and a validated Session Profile. Its final run JSON was
  reloaded through the production artifact validator.

Final representative batch:
`results/dcl-verification/20261107T062258_347e4e239532/`

Run JSON:
`363bc6983b8d4f1ba017109fe710ad9a.json`

Inspect these saved results with:

```bash
.venv/bin/streamlit run app.py -- --results results/dcl-verification
```

## Remaining limitations and operating notes

- **The full live DCL pipeline was not tested.** A later startup-check update
  successfully verified live key/model access using the existing `.env` with an
  eight-token probe (finish reason `length`). Injected/local-server DCL success
  verifies integration, persistence and validation; it does not establish a live
  model's interpretation quality, context capacity, latency or schema adherence.
- Ollama Cloud currently does not enforce structured-output schemas. Invalid JSON,
  unsupported evidence, omitted required coverage and invalid numerical checks
  fail explicitly; there is no fabricated-output fallback.
- Mathematical verification applies to structured claim values. Prose receives
  conservative screening, which cannot guarantee semantic correctness. Human
  review remains necessary; model confidence is not used as evidence.
- Missing metadata remains unresolved. Robustness conclusions apply only to the
  recorded interventions; omission effects are descriptive, not causal. Fixed
  applied experiment weights remain fixed during omission counterfactuals.
- Prompts exclude participant identifiers and names. Sanitization handles known
  credentials and recognizable sensitive fields, not every possible sensitive
  statement in arbitrary prose; source metadata still requires appropriate review.
- Completed attempts survive interruption in atomic sidecar files. An interrupted
  batch may require reading these directly because run/manifest persistence occurs
  separately. Historical results are not regenerated or overwritten.
- Fresh runs regenerate comprehension. Downstream task packages require matching
  run, source and configuration hashes. No partial regeneration scheduler or human
  review workflow was added.
- The suite emits existing Streamlit bare-mode and pandas deprecation warnings;
  they do not prevent the checks from passing.

## `.env` and startup-check update

- Declared `python-dotenv==1.2.4`; `AIConfig.from_env()` loads the explicit
  repository-root `.env` with `override=False`. Passing `env_file=None` disables
  file loading. Existing `.env` values were preserved; `.env.example` contains
  placeholders only.
- Added `src/preflight.py`. The CLI sends one fixed, non-streaming startup probe
  with eight output tokens, a 30-second timeout and SDK retries disabled. It
  accepts `stop` and `length` finishes without requiring an exact reply, rejects
  malformed/refused responses, and reports sanitized errors with exit code 2 for
  authentication/model/request configuration failures or 1 for inconclusive
  transport/provider/response failures.
- Updated `run_experiment.py` to check access before batch creation/calculation,
  reuse the same client for the batch, and close it on success or failure.
  `validate` stays offline. Direct runner calls retain their injectable behavior.
- Updated `src/runner.py` and `src/results.py` to save a successful probe once as
  immutable, sanitized `preflight.json`, associated with its batch.
- Updated `app.py` to display **Batch startup check** separately from experiment
  calls using the shared message/input/output/receipt renderer. Updated README,
  CLI fixtures, dashboard tests and isolated environment/probe tests.
- Live verification: the sandboxed attempt could not connect; the approved retry
  with network access **passed**, with finish reason `length`. No credential
  values were printed and no experiment batch was run by that live probe.
- Final verification for the startup-check update: **76 tests passed** in
  134.242 seconds, including localhost SDK/CLI integration and dashboard audit
  display. Compilation and `git diff --check` also passed. `.env` remains ignored
  by Git.

## Live experiment failure correction

The failing batch `results/20261107T072431_44ceee8f49d6` was inspected without
modification. The model returned JSON successfully, but session-understanding
validation rejected factual claims with null checks, uncertainty findings with
empty caveats, and invented subject IDs such as metadata-topic labels. The CLI
previously hid those reasons behind a generic stage-failure message.

Updated `src/models/prompt.py` with version `dcl-prompt-v2`, stage-specific rules,
explicit allowed subjects, valid checked/uncertain examples and output templates
populated with actual entity IDs, evidence references and exact checked values.
These templates are input guidance only: model output is never merged with them
or silently repaired. Unfilled placeholders fail. `src/models/explanation.py` now
documents these cross-field requirements in the schema. The pipeline contract
version remains `dcl-v1`; prompt versions are tracked independently.

Live verification then exposed context exhaustion during stakeholder profiling:
the original request consumed 125,802 input tokens and ended with a truncated
response. `src/diagnostics.py` now derives compact intervention/correlation
summaries and explicit robustness exceptions. `src/comprehension.py` selects these
for model context, preserving prior citations and keeping the full diagnostic
vectors in the saved evidence registry. All tested conditions, winner/rank/tie
exceptions, undefined results and minority correlation outcomes remain represented.
Repeated code/dependency hash facts remain audit provenance rather than model
input; request JSON is serialized compactly. Final live S01 profiling used 72,054
input tokens and completed normally.

`ComprehensionFailure` now includes the failed phase and rejection reasons in CLI
messages. Validators, failure/dependency policy and the prohibition on silent
output repair remain in force. Regression coverage includes captured live-invalid
finding shapes, valid prompt examples/templates, placeholder rejection, compact
context selection, prior-evidence retention, minority/undefined agreement outcomes,
and the exact S01 robustness winner-change exceptions.

Live results:

- Final S01: all three stages passed in
  `results/dcl-fix-verification/20261107T075348_390cbefe49ec`.
- S02_MAX and S02_MIN: all three stages passed for both cases in
  `results/dcl-fix-verification/20261107T074529_d629a488c066`.
- The final S01 narrative correctly identifies C1/C2/C4 removal and C3 × 1.2 as
  the tested winner-change conditions. Structured validation does not prove all
  generated prose; human review remains necessary.
- The entire 40-case batch was not rerun against the live provider. Its numerical,
  integration and dashboard regression coverage uses injected/local-server clients.

The original experiment command can be rerun to produce a fresh batch with the
new prompt/context version. Historical failed artifacts are retained for audit.

Final correction verification: **83 tests passed** in 119.291 seconds, including
all 40 variations with injected AI clients and the SDK/CLI/dashboard regression
suite. Focused comprehension tests, compilation and `git diff --check` passed.
All three successful live run artifacts were reloaded through the production
`RunResult` validator. No historical failed artifact was overwritten.

## Agreement/unanimity validation correction

The S01 response in `results/20261107T082103_e3752587f4fc` claimed that matching
group rankings implied “unanimous agreement with no minority deviation.” The
validator correctly rejected that inference. The original failed artifact was
preserved; no assertion was automatically repaired or accepted.

- `src/models/prompt.py`: prompt version `dcl-prompt-v3` explicitly distinguishes
  modeled group-level alignment from participant/population consensus. The
  session-profile agreement template retains a conditional assessment, preference
  uncertainty, a scope caveat, and evidence references for small-group and missing
  stakeholder-statement limitations.
- `src/models/explanation.py`: the agreement field documents this scope and the
  requirement to preserve its limitations in the output schema.
- `src/comprehension.py`: the conservative interpretive screen now includes
  “unanimity” and recognizes local explicit negation. A disclaimer elsewhere in a
  statement cannot exempt an unsupported positive assertion. This remains a
  screening heuristic, not semantic proof.
- `tests/test_comprehension.py` and
  `tests/fixtures/dcl_live_unanimity_claim.json`: regression coverage retains the
  exact rejected live finding and confirms it remains unsupported, exercises
  qualified statements and mixed negation/assertion clauses, and verifies the
  template's scope, caveats and evidence references.

Live verification with the configured provider passed S01 and dependent S06,
including all six comprehension calls, in:
`results/dcl-agreement-verification/20261107T083014_90f6c609706a`.
Both saved runs reloaded through `RunResult`. Their agreement findings remain
conditional and explicitly state that modeled alignment does not establish
participant unanimity, population consensus or absence of minority views.
The full 40-case batch was not rerun live. Genuine unsupported responses still
fail and block dependents according to the configured policy; no semantic retry
or model-output substitution was introduced.

Final agreement-correction verification: **86 tests passed** in 120.515 seconds,
including all 40 experiment variations with injected clients and the existing
SDK/CLI/dashboard checks. The 28 focused comprehension tests, compilation and
`git diff --check` also passed. Live S01 and dependent S06 passed without weakening
the rejection of the captured unsupported unanimity assertion.

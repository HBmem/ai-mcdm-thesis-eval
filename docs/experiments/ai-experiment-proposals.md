# Additive AI experiment proposals: S10–S13

These proposals retain the existing schema 3.0 numerical structure. Resources,
processing, stakeholders, cases, operations, assertions, comparisons, and numerical
result dependencies stay in their existing locations. No input envelope is added.
Only S10–S13, their AI settings/output flags, and S13's extra source resources are
introduced here. S00–S09 and the Python implementation are unchanged.

## Experiments

| Experiment | Numerical conditions | AI intervention | AI results |
| --- | --- | --- | --- |
| S10 | Uniform preferences, matching S00 | None | One baseline three-layer workflow per AI repetition |
| S11 | MAX and MIN, matching S01 | Different numerical evidence | A separate three-layer workflow for each case |
| S12 | Exactly S10's numerical inputs | Reporting-layer task replacement only | One three-layer workflow with possible downstream advisory effects |
| S13 | Baseline inputs plus a strictly dominated A6 | Falsified model-facing session score/rank | One workflow observing contradictions, explanations, and advice |

The three overall layers execute in this order:

1. `data_comprehension`: builds structured context from domain information and
   supplied MCDM evidence. Its internal stages include `session_understanding`,
   `stakeholder_profiles`, and `session_profile`.
2. `data_explanation_reporting`: produces explanations and reports from numerical
   evidence and preceding comprehension records.
3. `policy_decision_advisory`: uses available results, context, explanations, and
   robustness diagnostics to support deliberation. A qualified answer or a refusal
   to recommend under unresolved contradictions is a meaningful study outcome.

Case count is not expanded for prompt or provider comparisons. S11 has two cases
because MAX and MIN are distinct numerical conditions; S10, S12, and S13 each have
one. AI results should be compared across experiments afterward. Existing
`comparisons` and `assertions` remain numerical checks on genuine results; they do
not score AI responses or compare generated prose.

## AI settings and defaults

`defaults.pipeline: "ai-mcdm"` follows the proposed S10 spelling. Its intended
execution is MCDM first, then the selected AI layers, despite the label's order.
Require `defaults.ai.enabled: true` for this pipeline. Existing `mcdm` experiments
with AI disabled continue to represent numerical-only execution.

- `ai.repetitions` is the number of independent full AI workflows per calculated
  numerical result. It does not recalculate the numerical result or mean transport
  retries. `defaults.repetitions` retains its numerical-run meaning.
- `ai.layers` selects the ordered overall layers. Reject unknown names, duplicates,
  or unsupported prerequisite omissions rather than fabricating earlier outputs.
- `ai.provider: null` and `ai.model: null` inherit the externally configured
  provider and model. Explicit nonempty strings select a provider profile/model;
  provider profiles hold endpoints and credential environment-variable names, not
  credentials in experiment JSON.
- `ai.parameters: {}` inherits the baseline generation settings. Explicit entries
  override those settings; unsupported provider parameters must be rejected rather
  than silently dropped. Record the complete effective configuration.
- `ai.prompt_overrides: {}` uses the versioned default prompts. An override is keyed
  by overall layer and contains a `system` or `task` replacement. S12 replaces only
  `data_explanation_reporting.task`; it leaves DCL and advisory prompts unchanged.
  Omissions inherit default components. Reject unknown keys, nonstrings, and blank
  replacements. Default output schemas/protocol guidance remain program-owned.
- `ai.reported_result_interventions: []` leaves the evidence unmodified. These
  interventions operate on a model-facing result copy after genuine calculation
  and numerical validation, not on the MCDM engine or the input performance data.

One numerical result can therefore produce multiple AI repetitions. Each workflow
has fresh state; preceding-layer artifacts may be reused only within that same
workflow. S11/MAX never receives S11/MIN's response or vice versa. Provider/model
interventions can be separate experiment files copied from S10; hold other
settings fixed where possible. Changing the model and provider together measures
their combined effect. Numerical seeds do not establish AI determinism.

## S13 treatment and matched control

The extra files `alternative/transport_expanded_v1.json` and
`datasets/transport_expanded_v1.csv` retain the original five options and rows,
and add A6 with cost 12 million USD, accessibility 20, environment 10, and duration
36 months. Every existing alternative strictly dominates A6 under the declared
min/max criterion directions. Its legitimate TOPSIS score is zero with the current
baseline setup.

`promote_alternative` with scope `session` targets only the reported session result:

- Assign A6 reported score 1.0 and rank 1. Require its score to exceed every other
  reported score; reject a tied or invalid promotion rather than silently change
  an unrelated score.
- Preserve all other session scores and order those remaining alternatives by
  score, assigning ranks 2 onward with the program's normal tie presentation rule.
- Update every model-facing copy of the session winner, score/rank table, and
  derived summary to refer to this same reported result. Do not leave a stale
  winner label that creates a trivial duplicate-table discrepancy.
- Preserve raw criterion values, definitions/directions, preference judgments,
  weights, genuine intermediate numerical evidence, and all participant/group
  rankings. Contradictions with those data are intentional. Do not falsify those
  records to make the promoted result mathematically valid.
- Retain a separate immutable genuine calculation and an intervention audit. Never
  overwrite the numerical artifacts or rerun TOPSIS to justify the reported score.

All three AI layers receive the reported evidence consistently. Genuine numerical
validation happens before the intervention. Save raw AI outputs before later
response validation so spontaneous detection can be distinguished from detection
by programmatic checks. Do not silently repair the treatment before the AI sees
it. If the workflow stops after an unresolved contradiction, record that outcome
rather than force a policy recommendation.

S10 is the common study baseline, but S13 also changes the alternative set. To
isolate the reporting manipulation, run a separate matched-control experiment
copied from S13, with a distinct experiment ID and
`reported_result_interventions: []`. That control uses the exact same expanded
dataset and prompts. Do not attribute every S13-versus-S10 difference to tampering
alone. This proposal does not create an additional experiment ID without an
explicit choice for that companion experiment.

The A6 description is neutral. Keep researcher experiment/case names, purposes,
hypotheses, filenames/resource paths, intervention settings, and audit records out
of model context. Only allowlisted domain information and numerical evidence are
sent. Use neutral model-facing session identifiers so the experiment name cannot
reveal its condition. Instructions to evaluate the study belong outside prompts.

## Output and implementation status

`save_ai_calls` saves exact rendered role-labelled messages, evidence identifiers
and hashes, effective provider/model/settings, layer/stage, repetition, attempt,
and errors. `save_ai_responses` saves raw responses and parsed outputs. Existing
numerical output settings continue to save genuine numerical inputs and results.
Save the reported evidence copy with the AI call and preserve its linkage to the
original numerical result. Saving flags do not decide whether calls execute.
Never persist credential values. Keep numerical and AI statuses separate.

These are proposed configurations, not implemented AI execution. The current
loader/planner rejects AI-enabled experiments, AI models preserve only `enabled`,
and the executor rejects AI calls. Required focused additions are parsing these
AI fields, three-layer orchestration/default prompts, scoped task replacement,
provider/model resolution, reported-result copy interventions, and AI persistence.
The numerical runner's existing input format does not need to be reorganized.

# Schema 3.0: preference codes, participant generators, and case loading

The examples now use the production implementation. Run from the repository root:

```bash
.venv/bin/python -m docs.examples.experiment_loading
.venv/bin/python -m unittest docs.examples.test_experiment_loading -v
```

## Preference scale

Schema 3.0 now declares:

```json
"judgment_encoding": "pairwise_preference"
```

Judgments are legacy `PairwisePreference` codes, not ratios:

| Code | Current AHP ratio |
| --- | --- |
| -4 | 1/9 |
| -3 | 1/7 |
| -2 | 1/5 |
| -1 | 1/3 |
| 0 | 1 |
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |

JSON integers and canonical code strings (`0` or `"0"`) are accepted. Floats,
booleans, unknown codes, incomplete judgment sets, and unknown group IDs fail
validation. `parse_preference()` is the single parsing boundary.

Parsed and generated judgments retain enum members. Convert them into numerical
ratios at the calculation boundary using `preference.ahp_scale`. Generators copy
preferences without knowing their mathematical scale. Changing the ratio mapping
in that property does not require editing the loader or generation strategies.
Introducing a different enum/code vocabulary additionally requires updating the
parser and domain-model annotations; arbitrary scales are not automatically
supported by the current numerical engine. Record scale changes in provenance.

S01's neutral judgments are now code `0`. S05 originally used ratios such as 2
and 4 that the legacy scale cannot represent exactly. They were converted once
using nearest logarithmic distance, with ties favoring the intensity nearest 1.
The original ratios and conversion method are retained in
`stakeholders.scale_conversion`, and copied onto loaded experiments. This changes
S05's numerical inputs; do not compare its new scores with old scores as though
only the file encoding changed. The runtime loader never silently quantizes data.

## `_participants()` coordinates parsing and generation

The implementation lives in `src/experiments/participants.py` and is imported into
`src/experiments/loader.py`. It follows this sequence:

```python
# First build ExperimentParticipant objects from the predefined JSON entries.
# Each judgment is parsed with parse_preference(value).
validate_participants(
    participants, group_ids=group_ids, question_ids=question_ids,
)

# Then ask the generator for new participants only and append them.
participants.extend(generate_participants(
    generators,
    participants,
    group_ids=group_ids,
    question_ids=question_ids,
))
return participants
```

Both `_participants()` and `generate_participants()` return lists of your existing
`ExperimentParticipant` dataclass. The generator does not return the predefined
participants again. Its validation checks the combined IDs, so collisions with
predefined participants or previous generator rules are errors.

## Generator rule format

Put this inside `stakeholders.participants.generators`:

```json
{
  "generator_id": "additional_residents",
  "strategy": "repeat_templates",
  "group_id": "community_residents",
  "count": 4,
  "template_participant_ids": ["community_residents_p01"],
  "id_pattern": "{group_id}_{generator_id}_{index:04d}"
}
```

This adds four participants, giving five residents when one was predefined.
Templates must be predefined participants in the specified group. Multiple
specified templates are cycled in their declared order. Every generated person
receives a separate judgments dictionary.

Each strategy receives independent copies of the rule and predefined participants.
It cannot mutate caller inputs or use participants from earlier generator rules as
implicit templates. `repeat_templates` is deterministic and needs no seed.

## Adding a strategy

`generator_strategies()` returns a snapshot of registered strategy functions.
`register_generator_strategy()` adds a function without changing `_participants`
or its dispatcher. Duplicate strategy names fail explicitly.

For example, this complete handler creates new neutral participants without any
template participants:

```python
from src.experiments.models import ExperimentParticipant
from src.experiments.participants import register_generator_strategy
from src.mcdm.enums import PairwisePreference


@register_generator_strategy("uniform_preferences")
def uniform_preferences(rule, predefined):
    questions = rule["question_ids"]
    return [
        ExperimentParticipant(
            participant_id=f"{rule['group_id']}_{rule['generator_id']}_{index:04d}",
            group_id=rule["group_id"],
            judgments={qid: PairwisePreference.EQUAL_IMPORTANCE for qid in questions},
        )
        for index in range(1, rule["count"] + 1)
    ]
```

Corresponding rule:

```json
{
  "generator_id": "neutral_residents",
  "strategy": "uniform_preferences",
  "group_id": "community_residents",
  "count": 10,
  "question_ids": ["q1", "q2", "q3", "q4", "q5", "q6"]
}
```

Import the module containing your decorated handler once before loading experiments.
The strategy validates its own additional parameters; the dispatcher validates the
common fields and returned objects, including exact question coverage, enum
judgments, participant IDs, requested group, and requested count.

For randomized strategies, add a seed to the rule and use a local RNG. Keep the
actual generated participants in the saved run rather than relying on the seed
alone to reconstruct them.

## Integration into `load_experiment_json()`

The production function now calls `_participants()` after loading questions and
groups, using the actual JSON structure:

```python
config = stakeholder_data["participants"]
participants = _participants(
    predefined=config.get("predefined", []),
    generators=config.get("generators", []),
    label=data["experiment_id"],
    schema_version=data["schema_version"],
    group_ids={group.id for group in groups},
    question_ids={question.id for question in questions},
    judgment_encoding=stakeholder_data["judgment_encoding"],
)
```

It validates `combine=append` and `on_id_collision=error`, then passes the parsed
participants and groups to the case resolver. That resolver accepts `base_inputs`
so the resources and participant generators do not need to run a second time.

`load_experiment_json(path)` returns one `Experiment` per resolved case. The
function converts each experiment participant into your existing domain types:

```python
Participant(
    id=p.participant_id,
    name=p.participant_id,
    stakeholder_group_id=p.group_id,
    responses=[
        PairwiseResponse(question_id=qid, preference=preference)
        for qid, preference in p.judgments.items()
    ],
)
```

Defaults, case metadata, comparison references, assertions, encoding, and scale
conversion metadata remain available on the loaded `Experiment`.

## Variations and their generator use

`src/experiments/variations.py` resolves local `input_from` references and applies
operations in their listed order. Every case receives independent input copies;
case order does not affect inheritance resolution, and cycles fail validation.

`resize_group` uses the same generator registry, but its `count` means final group
size. It computes how many additional participants are needed and calls the
generator with that count. It allocates final IDs while skipping preserved IDs.
With `preserve_existing=true`, shrinking is rejected. With false, the group is
replaced with the requested number of generated participants.

The implemented operations are `resize_group` and `set_group_voting_powers`.
Other operations fail explicitly. Cross-experiment input inheritance, case-level
processing overrides, and AI pipelines still require their own implementations.

## Single-file loading versus batch loading

```python
from pathlib import Path
from src.experiments.loader import load_experiment_json, load_experiments

s05_cases = load_experiment_json(
    Path("experiments/S05_voting_power_and_participation_imbalance.json")
)
all_cases = load_experiments()  # Repository-relative default directory.
```

Single-file loading preserves external comparison references such as `S01/control`
without requiring those experiments to be loaded. Batch loading validates all
baseline/comparison/assertion references against the available cases. Neither
function calculates results, evaluates assertions, or calls an AI service. The
execution planner must expand repetitions and resolve result dependencies later.

Expected case sizes are 5 for S01, and 5, 5, and 50 for S05. Group replication in
S05 preserves the group's preference profile; repeating a heterogeneous template
list to a count that changes its proportions need not preserve its aggregate.

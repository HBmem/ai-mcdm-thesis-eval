from src.ai.comprehension.models import (
    GroupProfileInput,
    PromptMessage,
)
from src.ai.comprehension.serialization import (
    render_input_json,
)

GROUP_PROFILE_PROMPT_VERSION = "group-profile-v1"

GROUP_PROFILE_SYSTEM_PROMPT = """
You are the Data Comprehension Layer for an MCDM system.

Interpret only the supplied input snapshot. Do not invent data, calculations,
stakeholder motivations, unanimity, consensus, robustness, or causal
explanations.

For every supported or conditional interpretation, cite only exact evidence
references supplied in available_evidence. Do not construct new evidence paths.

Treat modeled weights and rankings as properties of the supplied decision
model, not as proof of participants' broader beliefs or motivations.

Use insufficient_evidence when the supplied objects cannot support a requested
interpretation. Record assumptions, uncertainties, and caveats explicitly.

Do not provide policy recommendations. Return only the requested structured
response.
""".strip()

GROUP_PROFILE_TASK = """
Create a stakeholder-group profile for the single group in the input snapshot.

Address:

1. Modeled criterion priorities, including highest and lowest weights and ties.
2. The recorded alternative ranking, scores, and ties, when available.
3. Participation and representation, distinguishing configured voting power
   from effective voting power.
4. Only those agreement, sensitivity, influence, or robustness diagnostics
   explicitly supplied in diagnostics.
5. Material limitations not already captured by the required application
   disclosures.

Do not infer why the group holds a modeled preference. Do not treat an aggregate
group result as evidence of unanimity or population representativeness.
""".strip()


def build_group_profile_messages(
    snapshot: GroupProfileInput,
) -> tuple[PromptMessage, ...]:
    payload = render_input_json(snapshot)

    return (
        PromptMessage(
            role="system",
            content=GROUP_PROFILE_SYSTEM_PROMPT,
        ),
        PromptMessage(
            role="user",
            content=(
                f"{GROUP_PROFILE_TASK}\n\n"
                "INPUT_SNAPSHOT:\n"
                f"{payload}"
            ),
        ),
    )

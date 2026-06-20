from typing import Dict

from decisiondesk.agents.base import complete_structured
from decisiondesk.core.schema import PlannerOutput
from decisiondesk.core.state import DecisionState

MODEL = "gpt-5-mini"

_SYSTEM = (
    "You are a principal software engineer running an architecture review. "
    "Your job is to decompose a binary technology decision into precisely targeted "
    "sub-questions for four specialist reviewers: scalability, security, cost, and "
    "maintainability. Each question must be specific to the two options being compared "
    "and draw out the most consequential tradeoffs in that dimension. "
    "Do not express a preference or recommendation. Respond only with valid JSON."
)

_PROMPT = """\
You are facilitating an architecture review for the following decision:

QUESTION: {question}

OPTION A: {option_a}
OPTION B: {option_b}

CONSTRAINTS:
{constraints}

Your task:

1. DECISION FRAMING — restate the decision in one precise sentence that clarifies
   exactly what is being compared and in what context (infer context from the constraints).

2. CONTEXT SUMMARY — 2–3 sentences on what makes this decision non-trivial.
   Name the real tension (e.g. operational simplicity vs query flexibility,
   consistency vs availability, build cost vs vendor lock-in).

3. SPECIALIST QUESTIONS — write one targeted question per specialist.
   Each question must:
   - Name both options explicitly
   - Ask about the dimension that specialist owns
   - Surface the sharpest tradeoff, not a generic concern
   - Be answerable with a concrete recommendation (option_a / option_b / neutral)

   scalability_question  → throughput, read/write patterns, sharding, horizontal scaling limits
   security_question     → auth boundaries, encryption, attack surface, compliance exposure
   cost_question         → infrastructure spend, licensing, egress fees, operational overhead, TCO
   maintainability_question → team familiarity, debugging story, migration risk, operational toil

4. KEY CONSTRAINTS — list 3–5 constraints extracted or inferred from the input
   that every specialist must respect (e.g. "team of 3 engineers",
   "budget under $500/mo", "must be GDPR-compliant", "expected 10k req/s at peak").

Respond with JSON matching this schema exactly:
{{
  "decision_framing": "...",
  "context_summary": "...",
  "scalability_question": "...",
  "security_question": "...",
  "cost_question": "...",
  "maintainability_question": "...",
  "key_constraints": ["...", "..."]
}}
"""


def run_planner(state: DecisionState) -> Dict:
    constraints_text = (
        "\n".join(f"- {k}: {v}" for k, v in state.input.constraints.items())
        if state.input.constraints
        else "None specified."
    )

    prompt = _PROMPT.format(
        question=state.input.question,
        option_a=state.input.option_a,
        option_b=state.input.option_b,
        constraints=constraints_text,
    )

    output = complete_structured(
        prompt=prompt,
        model=MODEL,
        output_schema=PlannerOutput,
        system=_SYSTEM,
    )

    return {"planner": output}

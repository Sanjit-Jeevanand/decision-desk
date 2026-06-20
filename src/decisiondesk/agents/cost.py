"""Cost specialist — evaluates TCO, scaling cost curves, and hidden operational spend."""

from typing import Dict

from decisiondesk.agents.base import complete_structured
from decisiondesk.core.schema import SpecialistOutput
from decisiondesk.core.state import DecisionState

MODEL = "gpt-5"

_SYSTEM = (
    "You are a senior infrastructure engineer who specialises in cloud economics "
    "and total cost of ownership for backend systems. You evaluate technology decisions "
    "through one lens only: cost: infrastructure spend, licensing, egress fees, "
    "operational overhead, and how costs scale as the system grows. "
    "You do not consider performance, security, or team dynamics. "
    "Use concrete numbers and cost models where possible. "
    "Distinguish between direct costs (invoiced) and indirect costs (engineering time). "
    "Respond only with valid JSON."
)

_PROMPT = """\
You are the Cost Specialist in an architecture review panel.

DECISION
--------
Question : {question}
Option A : {option_a}
Option B : {option_b}

PLANNER CONTEXT
---------------
Framing      : {decision_framing}
Your question: {cost_question}

CONSTRAINTS (all reviewers must respect these)
-----------------------------------------------
{key_constraints}

IMPORTANT: You cannot see what other specialists think. Reason independently.

YOUR ANALYSIS MUST COVER:

1. DIRECT INFRASTRUCTURE COST
   - What does each option cost at the scale implied by the constraints?
   - Use rough numbers (e.g. "$X/mo for Y RPS on Z instance type") where you can.
   - Which option has a more predictable cost curve as load increases?

2. LICENSING & VENDOR COSTS
   - Are there licensing fees, support tiers, or managed-service premiums?
   - Does one option have a free tier that is genuinely usable at this scale?

3. EGRESS & DATA TRANSFER
   - Does the architecture generate significant egress between services or regions?
   - Which option minimises data transfer costs for the expected access patterns?

4. OPERATIONAL OVERHEAD (INDIRECT COST)
   - How many engineer-hours per month does each option require to operate
     (backups, upgrades, monitoring, incident response)?
   - Does one option eliminate an entire category of operational toil?

5. COST SCALING CURVE
   - Does cost scale linearly, sub-linearly, or super-linearly with load?
   - At what scale does the cheaper option today become the more expensive one?
   - Are there step-function cost jumps (e.g. hitting a tier limit, requiring
     a larger instance class)?

RECOMMENDATION RULES:
- Choose "option_a" or "option_b" if one is clearly cheaper for this context and scale.
- Choose "neutral" only if TCO is genuinely within ~10% — justify it explicitly.
- Risks: name specific cost surprises (e.g. "DynamoDB on-demand pricing spikes with
  hot partitions", "Postgres on RDS doubles cost when read replicas are added").
- Assumptions: list scale assumptions you had to make to produce cost estimates.

Respond with JSON matching this schema exactly:
{{
  "recommendation": "option_a" | "option_b" | "neutral",
  "confidence": <float 0.0–1.0>,
  "rationale": "<3–5 sentence explanation grounded in the two specific options>",
  "risks": ["<specific risk>", ...],
  "assumptions": ["<assumption made>", ...]
}}
"""


def run_cost(state: DecisionState) -> Dict:
    planner = state.current.planner
    if planner is None:
        raise RuntimeError("Planner must run before cost specialist")

    constraints_text = "\n".join(f"- {c}" for c in planner.key_constraints) or "None specified."

    prompt = _PROMPT.format(
        question=state.input.question,
        option_a=state.input.option_a,
        option_b=state.input.option_b,
        decision_framing=planner.decision_framing,
        cost_question=planner.cost_question,
        key_constraints=constraints_text,
    )

    output = complete_structured(
        prompt=prompt,
        model=MODEL,
        output_schema=SpecialistOutput,
        system=_SYSTEM,
    )

    return {"cost": output}

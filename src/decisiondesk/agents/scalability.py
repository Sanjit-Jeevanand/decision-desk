"""Scalability specialist — evaluates throughput, scaling limits, and growth headroom."""

from typing import Dict

from decisiondesk.agents.base import complete_structured
from decisiondesk.core.schema import SpecialistOutput
from decisiondesk.core.state import DecisionState

MODEL = "gpt-5"

_SYSTEM = (
    "You are a senior distributed systems engineer with deep experience in backend "
    "scalability. You evaluate technology decisions through one lens only: "
    "how each option performs under load, scales horizontally, and holds up as "
    "data volume and traffic grow. You do not consider cost, security, or team "
    "dynamics — those are handled by other reviewers. Be direct, specific, and "
    "back every claim with a concrete mechanism (not a vague assertion). "
    "Respond only with valid JSON."
)

_PROMPT = """\
You are the Scalability Specialist in an architecture review panel.

DECISION
--------
Question : {question}
Option A : {option_a}
Option B : {option_b}

PLANNER CONTEXT
---------------
Framing      : {decision_framing}
Your question: {scalability_question}

CONSTRAINTS (all reviewers must respect these)
-----------------------------------------------
{key_constraints}

IMPORTANT: You cannot see what other specialists think. Reason independently.

YOUR ANALYSIS MUST COVER:

1. READ/WRITE THROUGHPUT
   - What are the realistic throughput ceilings for each option at the scale
     implied by the constraints?
   - Which option degrades more gracefully under sustained high load?

2. HORIZONTAL SCALING
   - How does each option scale out (add nodes, replicas, shards)?
   - Are there architectural limits that become painful before the team hits them
     (e.g. single-master writes, connection pool exhaustion, hot partitions)?

3. DATA VOLUME & GROWTH
   - How does query performance change as the dataset grows by 10× or 100×?
   - Does one option require re-architecture at a predictable scale threshold?

4. OPERATIONAL SCALING
   - What does scaling actually require at runtime — config change, schema migration,
     re-sharding, DNS update, provisioned throughput bump?
   - Which has the lower blast radius when scaling goes wrong?

RECOMMENDATION RULES:
- Choose "option_a" or "option_b" if one is clearly better for scalability.
- Choose "neutral" only if they are genuinely equivalent at the scale implied
  by the constraints — explain exactly why.
- Confidence must reflect genuine uncertainty. Do not round up.
- Risks: name specific failure modes, not categories.
- Assumptions: list what you had to assume that the constraints did not specify.

Respond with JSON matching this schema exactly:
{{
  "recommendation": "option_a" | "option_b" | "neutral",
  "confidence": <float 0.0–1.0>,
  "rationale": "<3–5 sentence explanation grounded in the two specific options>",
  "risks": ["<specific risk>", ...],
  "assumptions": ["<assumption made>", ...]
}}
"""


def run_scalability(state: DecisionState) -> Dict:
    planner = state.current.planner
    if planner is None:
        raise RuntimeError("Planner must run before scalability specialist")

    constraints_text = "\n".join(f"- {c}" for c in planner.key_constraints) or "None specified."

    prompt = _PROMPT.format(
        question=state.input.question,
        option_a=state.input.option_a,
        option_b=state.input.option_b,
        decision_framing=planner.decision_framing,
        scalability_question=planner.scalability_question,
        key_constraints=constraints_text,
    )

    output = complete_structured(
        prompt=prompt,
        model=MODEL,
        output_schema=SpecialistOutput,
        system=_SYSTEM,
    )

    return {"scalability": output}

"""Detector agent — surfaces conflicts between the 4 specialist outputs."""

from typing import Dict

from decisiondesk.agents.base import complete_structured
from decisiondesk.core.schema import DetectorOutput
from decisiondesk.core.state import DecisionState

MODEL = "gpt-5-mini"

_SYSTEM = (
    "You are a senior engineering lead running an architecture review. "
    "Your sole job is to identify where the four specialist reviewers disagree — "
    "on recommendations, on confidence, or on the risks they name. "
    "You do not form your own opinion on the decision. "
    "You surface conflicts so the critic and synthesizer can resolve them. "
    "A conflict is only blocking if reasonable engineers could not proceed without "
    "resolving it first. Respond only with valid JSON."
)

_PROMPT = """\
You are the Conflict Detector in an architecture review panel.

DECISION
--------
Question : {question}
Option A : {option_a}
Option B : {option_b}

SPECIALIST OUTPUTS
------------------
SCALABILITY
  Recommendation : {scalability_rec} (confidence: {scalability_conf:.0%})
  Rationale      : {scalability_rationale}
  Risks          : {scalability_risks}

SECURITY
  Recommendation : {security_rec} (confidence: {security_conf:.0%})
  Rationale      : {security_rationale}
  Risks          : {security_risks}

COST
  Recommendation : {cost_rec} (confidence: {cost_conf:.0%})
  Rationale      : {cost_rationale}
  Risks          : {cost_risks}

MAINTAINABILITY
  Recommendation : {maintainability_rec} (confidence: {maintainability_conf:.0%})
  Rationale      : {maintainability_rationale}
  Risks          : {maintainability_risks}

YOUR TASK — identify every conflict across three dimensions:

1. RECOMMENDATION CONFLICTS
   Which specialists disagree on option_a vs option_b?
   A split recommendation is significant — name who disagrees with whom and why.

2. CONFIDENCE CONFLICTS
   Are any specialists highly confident in opposite directions?
   Two specialists at 85%+ confidence pointing at different options is a hard conflict.
   Two specialists at 55% confidence pointing at different options is a soft conflict.

3. RISK CONTRADICTIONS
   Does one specialist's rationale directly contradict another's?
   (e.g. cost says "DynamoDB scales cheaply" but scalability says "DynamoDB costs
   explode with hot partitions" — that's a contradiction, not just a different view.)

CLASSIFICATION RULES:
- blocking: true  → specialists with high confidence (>70%) recommend different options,
                    OR a risk named by one specialist directly invalidates another's rationale.
- blocking: false → specialists agree on direction but differ on tradeoffs, OR conflicts
                    are between low-confidence assessments only.
- severity "high"   → 3+ conflicts OR blocking=true with high-confidence split
- severity "medium" → 1–2 conflicts, at least one meaningful
- severity "low"    → minor differences in risk framing only, no recommendation split

Each conflict in the list must have:
  - "agents": the two (or more) specialists involved
  - "dimension": "recommendation" | "confidence" | "risk_contradiction"
  - "description": one sentence naming exactly what contradicts what

Respond with JSON matching this schema exactly:
{{
  "has_conflicts": true | false,
  "conflicts": [
    {{
      "agents": ["specialist_a", "specialist_b"],
      "dimension": "recommendation" | "confidence" | "risk_contradiction",
      "description": "..."
    }}
  ],
  "blocking": true | false,
  "severity": "low" | "medium" | "high"
}}
"""


def _fmt_risks(risks: list) -> str:
    return "; ".join(risks) if risks else "none"


def run_detector(state: DecisionState) -> Dict:
    for name in ("scalability", "security", "cost", "maintainability"):
        if getattr(state.current, name) is None:
            raise RuntimeError(f"Specialist '{name}' must run before detector")

    sc = state.current.scalability
    se = state.current.security
    co = state.current.cost
    ma = state.current.maintainability

    prompt = _PROMPT.format(
        question=state.input.question,
        option_a=state.input.option_a,
        option_b=state.input.option_b,
        scalability_rec=sc.recommendation,
        scalability_conf=sc.confidence,
        scalability_rationale=sc.rationale,
        scalability_risks=_fmt_risks(sc.risks),
        security_rec=se.recommendation,
        security_conf=se.confidence,
        security_rationale=se.rationale,
        security_risks=_fmt_risks(se.risks),
        cost_rec=co.recommendation,
        cost_conf=co.confidence,
        cost_rationale=co.rationale,
        cost_risks=_fmt_risks(co.risks),
        maintainability_rec=ma.recommendation,
        maintainability_conf=ma.confidence,
        maintainability_rationale=ma.rationale,
        maintainability_risks=_fmt_risks(ma.risks),
    )

    output = complete_structured(
        prompt=prompt,
        model=MODEL,
        output_schema=DetectorOutput,
        system=_SYSTEM,
    )

    return {"detector": output}

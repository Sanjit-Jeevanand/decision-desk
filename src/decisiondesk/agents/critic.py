"""Critic agent — challenges assumptions and surfaces hidden risks before synthesis."""

from typing import Dict

from decisiondesk.agents.base import complete_structured
from decisiondesk.core.schema import CriticOutput
from decisiondesk.core.state import DecisionState

MODEL = "gpt-5-mini"

_SYSTEM = (
    "You are a staff engineer known for asking the questions no one else asks. "
    "You have seen architecture decisions go wrong because reviewers optimised for "
    "the known constraints and missed the hidden ones. "
    "Your job is not to re-evaluate the options — the specialists have done that. "
    "Your job is to challenge the assumptions they made, identify risks they collectively "
    "missed, and decide whether the analysis is solid enough to act on or needs revision. "
    "Be precise and actionable. Vague criticism is useless. "
    "Respond only with valid JSON."
)

# Revision mode: user has acknowledged issues — stop finding new ones, just clear the resolved ones.
_REVISION_SYSTEM = (
    "You are in revision mode of an architecture review. "
    "The team has already reviewed the first-pass critique and responded. "
    "Do NOT surface new issues. Your only job is to confirm which original issues "
    "remain critically blocking DESPITE the user's response, and set "
    "requires_revision: false so the synthesizer can produce a final answer. "
    "If the user's response is reasonable, clear the issue. "
    "Return an empty issues list if all concerns are adequately addressed. "
    "Respond only with valid JSON."
)

_REVISION_PROMPT = """\
REVISION PASS
Question : {question}
Option A : {option_a}
Option B : {option_b}

HUMAN REVIEW INPUT
------------------

TASK
----
Based on the user's acknowledged issues and additional context above:
- Clear any issue the user explicitly acknowledged or provided a reasonable response to.
- Return ONLY issues that remain critically blocking despite the user's input.
- Do NOT raise any new issues.
- Set requires_revision: false — this is the final review cycle.

Respond with JSON matching this schema exactly:
{{
  "issues": ["<issue still blocking after user input>", ...],
  "requires_revision": false,
  "revised_constraints": []
}}
"""

_PROMPT = """\
You are the Critic in an architecture review panel.

DECISION
--------
Question : {question}
Option A : {option_a}
Option B : {option_b}

PLANNER FRAMING
---------------
{decision_framing}
{context_summary}

Key constraints identified: {key_constraints}

SPECIALIST OUTPUTS (summarised)
--------------------------------
Scalability  → {scalability_rec} ({scalability_conf:.0%}) : {scalability_rationale}
Security     → {security_rec} ({security_conf:.0%}) : {security_rationale}
Cost         → {cost_rec} ({cost_conf:.0%}) : {cost_rationale}
Maintainability → {maintainability_rec} ({maintainability_conf:.0%}) : {maintainability_rationale}

DETECTOR FINDINGS
-----------------
Conflicts found : {has_conflicts}
Severity        : {severity}
Blocking        : {blocking}
Conflicts       : {conflicts_text}

YOUR TASK — three things:

1. CHALLENGE ASSUMPTIONS
   Each specialist made assumptions (they listed them). Find the ones that are:
   - Contradicted by the constraints
   - Unlikely given the context
   - Shared by multiple specialists (a collective blind spot)
   Name the assumption and why it is suspect.

2. SURFACE MISSED RISKS
   What has the entire panel failed to consider?
   Common gaps in backend architecture reviews:
   - Operational incidents: what happens when this breaks at 3am?
   - Organisational risk: is the option sustainable given realistic team turnover?
   - Reversibility: how hard is it to undo this decision in 12 months?
   - Dependency risk: does one option create a hidden dependency on a third party?
   - Regulatory drift: are there upcoming compliance changes that affect one option more?

3. REQUIRES REVISION?
   Set requires_revision: true if ANY of:
   - A specialist assumption contradicts the stated constraints
   - A risk exists that would change the recommendation if acknowledged
   - The detector flagged blocking=true AND the issues are not addressed in the rationales
   Set requires_revision: false if the issues you found are edge cases that
   do not change the overall direction.

4. REVISED CONSTRAINTS (only if requires_revision: true)
   If revision is needed, list the missing context as additional constraints.
   Format each as a string: "constraint_name: value"
   Leave as [] if requires_revision is false.

Respond with JSON matching this schema exactly:
{{
  "issues": ["<specific actionable issue>", ...],
  "requires_revision": true | false,
  "revised_constraints": ["constraint_name: value", ...]
}}
"""


def _fmt_conflicts(conflicts: list) -> str:
    if not conflicts:
        return "none"
    return "; ".join(
        f"{c.agents} — {c.description}"
        for c in conflicts
    )


def run_critic(state: DecisionState) -> Dict:
    for name in ("planner", "scalability", "security", "cost", "maintainability", "detector"):
        if getattr(state.current, name) is None:
            raise RuntimeError(f"'{name}' must run before critic")

    # --- Revision mode: user has responded; don't hunt for new issues ---
    if state.revision_notes:
        header, task = _REVISION_PROMPT.format(
            question=state.input.question,
            option_a=state.input.option_a,
            option_b=state.input.option_b,
        ).split("TASK\n----", 1)
        prompt = header + state.revision_notes + "\n\nTASK\n----" + task
        return {"critic": complete_structured(
            prompt=prompt,
            model=MODEL,
            output_schema=CriticOutput,
            system=_REVISION_SYSTEM,
        )}

    # --- First pass: challenge assumptions and surface missed risks ---
    pl = state.current.planner
    sc = state.current.scalability
    se = state.current.security
    co = state.current.cost
    ma = state.current.maintainability
    de = state.current.detector

    prompt = _PROMPT.format(
        question=state.input.question,
        option_a=state.input.option_a,
        option_b=state.input.option_b,
        decision_framing=pl.decision_framing,
        context_summary=pl.context_summary,
        key_constraints=", ".join(pl.key_constraints) or "none",
        scalability_rec=sc.recommendation,
        scalability_conf=sc.confidence,
        scalability_rationale=sc.rationale,
        security_rec=se.recommendation,
        security_conf=se.confidence,
        security_rationale=se.rationale,
        cost_rec=co.recommendation,
        cost_conf=co.confidence,
        cost_rationale=co.rationale,
        maintainability_rec=ma.recommendation,
        maintainability_conf=ma.confidence,
        maintainability_rationale=ma.rationale,
        has_conflicts=de.has_conflicts,
        severity=de.severity,
        blocking=de.blocking,
        conflicts_text=_fmt_conflicts(de.conflicts),
    )

    return {"critic": complete_structured(
        prompt=prompt,
        model=MODEL,
        output_schema=CriticOutput,
        system=_SYSTEM,
    )}

"""Synthesizer agent — integrates all prior outputs into a final recommendation."""

from typing import Dict

from decisiondesk.agents.base import complete_structured
from decisiondesk.core.schema import SynthesizerOutput
from decisiondesk.core.state import DecisionState

MODEL = "gpt-5.5"

_SYSTEM = (
    "You are a principal engineer delivering the final verdict of an architecture review. "
    "You have the outputs of four specialist reviewers, a conflict detector, and a critic. "
    "Your job is to weigh all of this and make a single, decisive recommendation — "
    "not a hedge, not a 'it depends'. If the evidence supports a clear direction, "
    "commit to it and explain why. If genuine uncertainty remains after the full review, "
    "name exactly what would resolve it. Do not rehash everything the specialists said. "
    "Synthesise, don't summarise. Respond only with valid JSON."
)

_PROMPT = """\
You are the Synthesizer in an architecture review panel. You have the full picture.

DECISION
--------
Question : {question}
Option A : {option_a}
Option B : {option_b}

PLANNER FRAMING
---------------
{decision_framing}
{context_summary}

SPECIALIST RECOMMENDATIONS
---------------------------
Scalability     → {scalability_rec} ({scalability_conf:.0%})
Security        → {security_rec} ({security_conf:.0%})
Cost            → {cost_rec} ({cost_conf:.0%})
Maintainability → {maintainability_rec} ({maintainability_conf:.0%})

SPECIALIST RATIONALES (full)
-----------------------------
Scalability     : {scalability_rationale}
Security        : {security_rationale}
Cost            : {cost_rationale}
Maintainability : {maintainability_rationale}

CONFLICTS DETECTED
------------------
Has conflicts : {has_conflicts}
Severity      : {severity}
Blocking      : {blocking}
Details       : {conflicts_text}

CRITIC FINDINGS
---------------
Issues raised      : {issues}
Requires revision  : {requires_revision}
Revised constraints: {revised_constraints}

YOUR TASK:

1. WEIGH THE EVIDENCE
   - Which specialists carry the most weight for this specific decision context?
     (e.g. if the constraints imply a cost-sensitive startup, cost > scalability)
   - If specialists disagree, explain whose reasoning is more applicable here and why.
   - If blocking conflicts exist, resolve them or explain why they do not change the outcome.

2. FINAL RECOMMENDATION
   - Pick "option_a", "option_b", or "hybrid".
   - Use "hybrid" only when the evidence clearly shows that both options serve
     distinct, non-overlapping requirements (e.g. PostgreSQL for transactional data,
     DynamoDB for high-throughput event logs). If the two options are alternatives
     for the same role, pick one.
   - Do not output "neutral". If it is a coin-flip, pick the lower cost-of-being-wrong
     option and state that explicitly.

3. CONFIDENCE
   - Reflect the genuine confidence of the panel as a whole.
   - Reduce confidence if: blocking conflicts exist, critic flagged revision needed,
     or multiple specialists were below 65% confidence.
   - Do not inflate confidence to sound decisive.

4. RATIONALE (3–5 sentences)
   - Open with the single strongest reason for your recommendation.
   - Address the most significant conflict or risk raised by the panel.
   - Close with what would have changed the recommendation.

5. TRADEOFFS (3–5 items)
   - What does the chosen option give up?
   - Be specific — not "less flexible" but "DynamoDB's key schema makes it
     painful to support ad-hoc reporting queries added post-launch".

6. UNRESOLVED RISKS (0–3 items)
   - Risks that remain true regardless of the recommendation and that the
     team must actively manage.
   - If none, return an empty list.

Respond with JSON matching this schema exactly:
{{
  "final_recommendation": "option_a" | "option_b" | "hybrid",
  "confidence": <float 0.0–1.0>,
  "rationale": "<3–5 sentences>",
  "tradeoffs": ["<specific tradeoff>", ...],
  "unresolved_risks": ["<risk>", ...]
}}
"""


def _fmt_issues(issues: list) -> str:
    return "; ".join(issues) if issues else "none"


def _fmt_conflicts(conflicts: list) -> str:
    if not conflicts:
        return "none"
    return "; ".join(
        f"{c.agents} — {c.description}"
        for c in conflicts
    )


def run_synthesizer(state: DecisionState) -> Dict:
    for name in ("planner", "scalability", "security", "cost", "maintainability",
                 "detector", "critic"):
        if getattr(state.current, name) is None:
            raise RuntimeError(f"'{name}' must run before synthesizer")

    pl = state.current.planner
    sc = state.current.scalability
    se = state.current.security
    co = state.current.cost
    ma = state.current.maintainability
    de = state.current.detector
    cr = state.current.critic

    prompt = _PROMPT.format(
        question=state.input.question,
        option_a=state.input.option_a,
        option_b=state.input.option_b,
        decision_framing=pl.decision_framing,
        context_summary=pl.context_summary,
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
        issues=_fmt_issues(cr.issues),
        requires_revision=cr.requires_revision,
        revised_constraints=cr.revised_constraints or "none",
    )

    # Revision context appended outside .format() — user text may contain {}
    if state.revision_notes:
        prompt += (
            "\n\nHUMAN REVISION CONTEXT\n"
            "----------------------\n"
            "The user reviewed the first-pass critique and provided the following input.\n"
            "Weight this heavily — the user has accepted these risks and added context.\n"
            "Give a decisive recommendation: option_a, option_b, or hybrid.\n\n"
            + state.revision_notes
        )

    output = complete_structured(
        prompt=prompt,
        model=MODEL,
        output_schema=SynthesizerOutput,
        system=_SYSTEM,
    )

    return {"synthesizer": output}

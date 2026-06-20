"""Pydantic schemas for agent inputs and outputs."""

from typing import List, Dict
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Decision input
# ---------------------------------------------------------------------------

class ReviewInput(BaseModel):
    question: str
    option_a: str
    option_b: str
    constraints: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Agent outputs
# ---------------------------------------------------------------------------

class PlannerOutput(BaseModel):
    decision_framing: str           # one sentence restating the decision precisely
    context_summary: str            # what makes this decision non-trivial
    scalability_question: str       # targeted question for the scalability specialist
    security_question: str          # targeted question for the security specialist
    cost_question: str              # targeted question for the cost specialist
    maintainability_question: str   # targeted question for the maintainability specialist
    key_constraints: List[str]      # constraints all agents must respect


class SpecialistOutput(BaseModel):
    recommendation: str          # "option_a" | "option_b" | "neutral"
    confidence: float            # 0.0 – 1.0
    rationale: str
    risks: List[str]
    assumptions: List[str]


class ConflictItem(BaseModel):
    agents: List[str]
    dimension: str               # "recommendation" | "confidence" | "risk_contradiction"
    description: str


class DetectorOutput(BaseModel):
    has_conflicts: bool
    conflicts: List[ConflictItem]
    blocking: bool
    severity: str                # "low" | "medium" | "high"


class CriticOutput(BaseModel):
    issues: List[str]
    requires_revision: bool
    revised_constraints: List[str]  # ["constraint_name: revised value", ...]


class SynthesizerOutput(BaseModel):
    final_recommendation: str    # "option_a" | "option_b"
    confidence: float
    rationale: str
    tradeoffs: List[str]
    unresolved_risks: List[str]


class GateDecision(BaseModel):
    approved: bool
    reason: str
    tier: str


class GateOutput(BaseModel):
    decision: GateDecision

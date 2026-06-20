"""DecisionState — shared state threaded through the agent pipeline."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from decisiondesk.core.schema import (
    ReviewInput,
    PlannerOutput,
    SpecialistOutput,
    DetectorOutput,
    CriticOutput,
    SynthesizerOutput,
    GateOutput,
)


class GateTier(str, Enum):
    EXPLORATION = "exploration"  
    COMMITMENT  = "commitment"    
    OVERRIDE    = "override"    


class CurrentIteration(BaseModel):
    iteration: int = 1
    planner:        Optional[PlannerOutput]      = None
    scalability:    Optional[SpecialistOutput]   = None
    security:       Optional[SpecialistOutput]   = None
    cost:           Optional[SpecialistOutput]   = None
    maintainability: Optional[SpecialistOutput]  = None
    detector:       Optional[DetectorOutput]     = None
    critic:         Optional[CriticOutput]       = None
    synthesizer:    Optional[SynthesizerOutput]  = None
    gate:           Optional[GateOutput]         = None


class DecisionState(BaseModel):
    input: ReviewInput
    iteration: int = 1
    gate_tier: GateTier = GateTier.EXPLORATION
    force_approve: bool = False
    revision_notes: Optional[str] = None   # set during human-in-the-loop revision pass
    current: CurrentIteration = Field(default_factory=lambda: CurrentIteration(iteration=1))

    def authority_frozen(self) -> bool:
        return self.gate_tier in (GateTier.COMMITMENT, GateTier.OVERRIDE)

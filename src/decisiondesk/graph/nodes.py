from typing import Optional
from typing_extensions import TypedDict

from decisiondesk.core.schema import (
    PlannerOutput, SpecialistOutput, DetectorOutput,
    CriticOutput, SynthesizerOutput, GateOutput, ReviewInput,
)
from decisiondesk.core.state import CurrentIteration, DecisionState, GateTier
from decisiondesk.agents.planner import run_planner
from decisiondesk.agents.scalability import run_scalability
from decisiondesk.agents.security import run_security
from decisiondesk.agents.cost import run_cost
from decisiondesk.agents.maintainability import run_maintainability
from decisiondesk.agents.detector import run_detector
from decisiondesk.agents.critic import run_critic
from decisiondesk.agents.synthesizer import run_synthesizer
from decisiondesk.agents.gate import run_gate


# ---------------------------------------------------------------------------
# Flat graph state — one key per agent output
# ---------------------------------------------------------------------------

class LGState(TypedDict, total=False):
    # Input fields (required at graph entry)
    question:     str
    option_a:     str
    option_b:     str
    constraints:  dict

    # Execution control
    iteration:      int
    gate_tier:      str
    force_approve:  bool
    revision_notes: Optional[str]   # set during revision pass

    # Agent outputs (populated as the graph runs)
    planner:         Optional[PlannerOutput]
    scalability:     Optional[SpecialistOutput]
    security:        Optional[SpecialistOutput]
    cost:            Optional[SpecialistOutput]
    maintainability: Optional[SpecialistOutput]
    detector:        Optional[DetectorOutput]
    critic:          Optional[CriticOutput]
    synthesizer:     Optional[SynthesizerOutput]
    gate:            Optional[GateOutput]


# ---------------------------------------------------------------------------
# Conversion helper: LGState → DecisionState
# ---------------------------------------------------------------------------

def _to_ds(state: LGState) -> DecisionState:
    """Assemble a DecisionState from the flat LangGraph state dict."""
    return DecisionState(
        input=ReviewInput(
            question=state["question"],
            option_a=state["option_a"],
            option_b=state["option_b"],
            constraints=state.get("constraints", {}),
        ),
        iteration=state.get("iteration", 1),
        gate_tier=GateTier(state.get("gate_tier", GateTier.EXPLORATION.value)),
        force_approve=state.get("force_approve", False),
        revision_notes=state.get("revision_notes"),
        current=CurrentIteration(
            iteration=state.get("iteration", 1),
            planner=state.get("planner"),
            scalability=state.get("scalability"),
            security=state.get("security"),
            cost=state.get("cost"),
            maintainability=state.get("maintainability"),
            detector=state.get("detector"),
            critic=state.get("critic"),
            synthesizer=state.get("synthesizer"),
            gate=state.get("gate"),
        ),
    )


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def planner_node(state: LGState) -> dict:
    return run_planner(_to_ds(state))                          # {"planner": PlannerOutput}


def scalability_node(state: LGState) -> dict:
    return run_scalability(_to_ds(state))                      # {"scalability": SpecialistOutput}


def security_node(state: LGState) -> dict:
    return run_security(_to_ds(state))                         # {"security": SpecialistOutput}


def cost_node(state: LGState) -> dict:
    return run_cost(_to_ds(state))                             # {"cost": SpecialistOutput}


def maintainability_node(state: LGState) -> dict:
    return run_maintainability(_to_ds(state))                  # {"maintainability": SpecialistOutput}


def detector_node(state: LGState) -> dict:
    return run_detector(_to_ds(state))                         # {"detector": DetectorOutput}


def critic_node(state: LGState) -> dict:
    return run_critic(_to_ds(state))                           # {"critic": CriticOutput}


def synthesizer_node(state: LGState) -> dict:
    return run_synthesizer(_to_ds(state))                      # {"synthesizer": SynthesizerOutput}


def gate_node(state: LGState) -> dict:
    gate_output = run_gate(_to_ds(state))
    return {"gate": gate_output}                               # {"gate": GateOutput}

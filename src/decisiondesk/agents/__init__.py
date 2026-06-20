"""Agent runners. Each returns {agent_name: OutputModel}."""

from decisiondesk.agents.planner import run_planner
from decisiondesk.agents.scalability import run_scalability
from decisiondesk.agents.security import run_security
from decisiondesk.agents.cost import run_cost
from decisiondesk.agents.maintainability import run_maintainability
from decisiondesk.agents.detector import run_detector
from decisiondesk.agents.critic import run_critic
from decisiondesk.agents.synthesizer import run_synthesizer
from decisiondesk.agents.gate import run_gate

__all__ = [
    "run_planner",
    "run_scalability",
    "run_security",
    "run_cost",
    "run_maintainability",
    "run_detector",
    "run_critic",
    "run_synthesizer",
    "run_gate",
]

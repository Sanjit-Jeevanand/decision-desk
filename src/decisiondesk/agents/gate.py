"""Gate agent — pure rule-based approval logic. No LLM call."""

from decisiondesk.core.schema import GateDecision, GateOutput
from decisiondesk.core.state import DecisionState, GateTier


def run_gate(state: DecisionState) -> GateOutput:
    """
    Approval rules by tier:

    OVERRIDE    → always approved (force_approve flag set by user)
    COMMITMENT  → always approved (iteration 2+; human has reviewed and provided feedback)
    EXPLORATION → approved only if critic does not require revision AND
                  detector did not flag a blocking conflict
    """
    if state.force_approve or state.gate_tier == GateTier.OVERRIDE:
        return GateOutput(decision=GateDecision(
            approved=True,
            reason="Force-approved by override.",
            tier=GateTier.OVERRIDE.value,
        ))

    if state.gate_tier == GateTier.COMMITMENT:
        return GateOutput(decision=GateDecision(
            approved=True,
            reason="Commitment tier: human feedback incorporated, approval automatic.",
            tier=GateTier.COMMITMENT.value,
        ))

    # EXPLORATION tier — critic and detector both have veto power
    critic = state.current.critic
    detector = state.current.detector

    if critic is None:
        raise RuntimeError("Critic must run before gate")
    if detector is None:
        raise RuntimeError("Detector must run before gate")

    if critic.requires_revision:
        return GateOutput(decision=GateDecision(
            approved=False,
            reason="Critic flagged issues requiring revision before a decision can be approved.",
            tier=GateTier.EXPLORATION.value,
        ))

    if detector.blocking:
        return GateOutput(decision=GateDecision(
            approved=False,
            reason=f"Detector flagged blocking conflicts (severity: {detector.severity}). "
                   "Resolve conflicts before approving.",
            tier=GateTier.EXPLORATION.value,
        ))

    return GateOutput(decision=GateDecision(
        approved=True,
        reason="No blocking conflicts and no revision required. Decision approved.",
        tier=GateTier.EXPLORATION.value,
    ))

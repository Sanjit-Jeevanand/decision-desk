"""Revision graph — re-runs synthesizer → gate after the user has reviewed critic issues.

The critic is skipped: the user's acknowledgement IS the critic pass.
The caller pre-populates a cleared CriticOutput and sets force_approve=True.
"""

from langgraph.graph import END, StateGraph

from decisiondesk.graph.nodes import (
    LGState,
    synthesizer_node,
    gate_node,
)


def build_revision_graph():
    g = StateGraph(LGState)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("gate",        gate_node)

    g.set_entry_point("synthesizer")
    g.add_edge("synthesizer", "gate")
    g.add_edge("gate",        END)

    return g.compile()


revision_graph = build_revision_graph()

from langgraph.graph import END, StateGraph

from decisiondesk.graph.nodes import (
    LGState,
    cost_node,
    critic_node,
    detector_node,
    gate_node,
    maintainability_node,
    planner_node,
    scalability_node,
    security_node,
    synthesizer_node,
)


def build_parallel_graph():
    g = StateGraph(LGState)

    g.add_node("planner",         planner_node)
    g.add_node("scalability",     scalability_node)
    g.add_node("security",        security_node)
    g.add_node("cost",            cost_node)
    g.add_node("maintainability", maintainability_node)
    g.add_node("detector",        detector_node)
    g.add_node("critic",          critic_node)
    g.add_node("synthesizer",     synthesizer_node)
    g.add_node("gate",            gate_node)

    g.set_entry_point("planner")

    # Fan-out: planner → all 4 specialists (parallel)
    g.add_edge("planner", "scalability")
    g.add_edge("planner", "security")
    g.add_edge("planner", "cost")
    g.add_edge("planner", "maintainability")

    # Fan-in: all 4 specialists → detector (detector waits for all 4)
    g.add_edge("scalability",     "detector")
    g.add_edge("security",        "detector")
    g.add_edge("cost",            "detector")
    g.add_edge("maintainability", "detector")

    # Sequential chain after fan-in
    g.add_edge("detector",    "critic")
    g.add_edge("critic",      "synthesizer")
    g.add_edge("synthesizer", "gate")
    g.add_edge("gate",        END)

    return g.compile()


parallel_graph = build_parallel_graph()

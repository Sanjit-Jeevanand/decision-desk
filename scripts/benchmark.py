#!/usr/bin/env python3
"""
scripts/benchmark.py — parallel vs sequential TTA across 5 architecture decisions.

Method: runs each decision ONCE through the parallel graph, timestamps each node
as it completes via astream, then derives theoretical sequential time by summing
the 4 specialist durations (vs the actual max they took in parallel).

Usage (from decision-desk/):
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/benchmark.py

Cost: 5 runs × 8 LLM calls = 40 API calls (vs 80 for running both graphs).
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decisiondesk.core.state import GateTier
from decisiondesk.graph.nodes import LGState
from decisiondesk.graph.parallel import parallel_graph

# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

DECISIONS = [
    dict(
        label="DB: PostgreSQL vs DynamoDB",
        question="Should we use PostgreSQL or DynamoDB for our main application database?",
        option_a="PostgreSQL (RDS)",
        option_b="DynamoDB",
        constraints={"team_size": "medium", "traffic": "medium", "experience": "mixed", "budget": "medium"},
    ),
    dict(
        label="Cache: Redis vs Memcached",
        question="Should we use Redis or Memcached as our caching layer?",
        option_a="Redis",
        option_b="Memcached",
        constraints={"team_size": "small", "traffic": "high", "experience": "senior", "budget": "medium"},
    ),
    dict(
        label="Queue: Kafka vs RabbitMQ",
        question="Should we use Kafka or RabbitMQ for event streaming?",
        option_a="Kafka",
        option_b="RabbitMQ",
        constraints={"team_size": "medium", "traffic": "high", "experience": "mixed", "budget": "high"},
    ),
    dict(
        label="Arch: Microservices vs Monolith",
        question="Should we build the new product as microservices or a monolith?",
        option_a="Microservices",
        option_b="Monolith",
        constraints={"team_size": "small", "traffic": "low", "experience": "junior", "budget": "low"},
    ),
    dict(
        label="API: REST vs GraphQL",
        question="Should we expose our API as REST or GraphQL?",
        option_a="REST",
        option_b="GraphQL",
        constraints={"team_size": "medium", "traffic": "medium", "experience": "mixed", "budget": "medium"},
    ),
]

SPECIALISTS = {"scalability", "security", "cost", "maintainability"}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    label: str
    node_times: Dict[str, float] = field(default_factory=dict)   # node → wall seconds
    graph_wall_s: float = 0.0                                     # actual end-to-end time

    @property
    def specialist_parallel_s(self) -> float:
        """Wall time of the parallel phase = max of 4 specialist durations."""
        times = [self.node_times.get(s, 0) for s in SPECIALISTS]
        return max(times) if times else 0.0

    @property
    def specialist_sequential_s(self) -> float:
        """Theoretical time if specialists had run one after another."""
        return sum(self.node_times.get(s, 0) for s in SPECIALISTS)

    @property
    def sequential_total_s(self) -> float:
        """Theoretical total if ALL nodes ran sequentially."""
        return sum(self.node_times.values()) - self.specialist_parallel_s + self.specialist_sequential_s

    @property
    def parallel_total_s(self) -> float:
        return self.graph_wall_s

    @property
    def speedup(self) -> float:
        return self.sequential_total_s / self.parallel_total_s if self.parallel_total_s else 0.0

    @property
    def time_saved_s(self) -> float:
        return self.sequential_total_s - self.parallel_total_s


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_decision(d: dict) -> RunResult:
    state = LGState(
        question=d["question"],
        option_a=d["option_a"],
        option_b=d["option_b"],
        constraints=d.get("constraints", {}),
        iteration=1,
        gate_tier=GateTier.EXPLORATION.value,
        force_approve=False,
    )

    result   = RunResult(label=d["label"])
    t_graph  = time.perf_counter()
    # Track when specialists started (= when planner finished)
    t_planner_done: Optional[float] = None
    specialist_seen: set = set()

    async for chunk in parallel_graph.astream(state, stream_mode="updates"):
        t_now = time.perf_counter()
        for node_name, node_update in chunk.items():
            if node_name == "planner":
                # planner duration = time since graph start
                result.node_times["planner"] = t_now - t_graph
                t_planner_done = t_now

            elif node_name in SPECIALISTS:
                # specialist duration = time since planner finished (they all start together)
                specialist_seen.add(node_name)
                if t_planner_done is not None:
                    result.node_times[node_name] = t_now - t_planner_done

                # Detector starts when the LAST specialist finishes
                if specialist_seen == SPECIALISTS:
                    t_detector_start = t_now

            elif node_name == "detector":
                result.node_times["detector"] = t_now - t_detector_start if specialist_seen == SPECIALISTS else 0.0
                t_detector_done = t_now

            elif node_name == "critic":
                result.node_times["critic"] = t_now - t_detector_done
                t_critic_done = t_now

            elif node_name == "synthesizer":
                result.node_times["synthesizer"] = t_now - t_critic_done
                t_synthesizer_done = t_now

            elif node_name == "gate":
                result.node_times["gate"] = t_now - t_synthesizer_done

    result.graph_wall_s = time.perf_counter() - t_graph
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: List[RunResult]) -> None:
    W = 76
    BAR = 22

    def bar(s: float, max_s: float) -> str:
        filled = round(BAR * s / max_s) if max_s else 0
        return "█" * filled + "░" * (BAR - filled)

    print()
    print("=" * W)
    print("  DECISIONDESK — PARALLEL vs SEQUENTIAL TTA BENCHMARK")
    print("=" * W)

    for r in results:
        print(f"\n  {r.label}")
        print(f"  {'─' * (W - 2)}")
        max_t = max(r.node_times.values(), default=1)
        for node, t in r.node_times.items():
            tag = " (parallel × 4)" if node == "scalability" else \
                  " (↑ runs concurrently)" if node in SPECIALISTS else ""
            print(f"  {'✓':>2}  {node:<18} {bar(t, max_t)}  {t:5.2f}s{tag}")

        print()
        print(f"  Specialist phase  →  parallel {r.specialist_parallel_s:.2f}s  |  sequential (sum) {r.specialist_sequential_s:.2f}s")
        print(f"  Total             →  parallel {r.parallel_total_s:.2f}s  |  sequential (est.) {r.sequential_total_s:.2f}s")
        print(f"  Speedup: {r.speedup:.2f}×   Time saved: {r.time_saved_s:.1f}s")

    print()
    print("=" * W)
    print(f"  {'Decision':<36}  {'Parallel':>9}  {'Sequential':>11}  {'Speedup':>8}")
    print("  " + "─" * (W - 2))
    for r in results:
        print(f"  {r.label:<36}  {r.parallel_total_s:>8.1f}s  {r.sequential_total_s:>10.1f}s  {r.speedup:>7.2f}×")

    avg_p = sum(r.parallel_total_s   for r in results) / len(results)
    avg_s = sum(r.sequential_total_s for r in results) / len(results)
    avg_x = sum(r.speedup            for r in results) / len(results)
    overall_x = avg_s / avg_p

    print("  " + "═" * (W - 2))
    print(f"  {'MEAN':<36}  {avg_p:>8.1f}s  {avg_s:>10.1f}s  {overall_x:>7.2f}×")
    print("=" * W)
    print()
    print(f"  Mean TTA (parallel)          {avg_p:.1f}s")
    print(f"  Mean TTA (sequential est.)   {avg_s:.1f}s")
    print(f"  Speedup                      {overall_x:.2f}× (range: {min(r.speedup for r in results):.2f}–{max(r.speedup for r in results):.2f}×)")
    print()
    print("  ─" * (W // 2))
    print("  Resume bullet fill-in:")
    print()
    print(f'  "...parallel execution reduced wall-time {overall_x:.1f}× vs sequential')
    print(f'   (mean TTA {avg_p:.0f}s vs {avg_s:.0f}s across 5 architecture decisions)"')
    print()
    print("=" * W)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.\n  export OPENAI_API_KEY=sk-...", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Running {len(DECISIONS)} decisions through the parallel graph...")
    print(f"  ({len(DECISIONS) * 8} LLM calls total — ~5–10 min)\n")

    results: List[RunResult] = []

    for i, d in enumerate(DECISIONS, 1):
        print(f"  [{i}/{len(DECISIONS)}] {d['label']} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        r = await run_decision(d)
        results.append(r)
        print(f"{r.graph_wall_s:.1f}s  (speedup {r.speedup:.2f}×)")

    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())

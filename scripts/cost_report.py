#!/usr/bin/env python3
import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decisiondesk.agents.base import pop_usage, reset_usage, UsageRecord
from decisiondesk.core.state import GateTier
from decisiondesk.graph.nodes import LGState
from decisiondesk.graph.parallel import parallel_graph

# ---------------------------------------------------------------------------
# Pricing  (USD per 1M tokens)
# ---------------------------------------------------------------------------

PRICING = {
    #  model              input    output   (USD per 1M tokens)
    "gpt-5.5":        (   5.00,   30.00),
    "gpt-5":          (   1.25,   10.00),
    "gpt-5-mini":     (   0.25,    2.00),
}

NAIVE_MODEL = "gpt-5.5"   # "use GPT-5.5 everywhere" baseline

# ---------------------------------------------------------------------------
# Decisions (diverse so rejection rate is meaningful)
# ---------------------------------------------------------------------------

DECISIONS = [
    dict(
        question="Should we use PostgreSQL or DynamoDB for our main application database?",
        option_a="PostgreSQL (RDS)", option_b="DynamoDB",
        constraints={"team_size": "medium", "traffic": "medium", "experience": "mixed", "budget": "medium"},
    ),
    dict(
        question="Should we use Redis or Memcached as our caching layer?",
        option_a="Redis", option_b="Memcached",
        constraints={"team_size": "small", "traffic": "high", "experience": "senior", "budget": "medium"},
    ),
    dict(
        question="Should we use Kafka or RabbitMQ for event streaming?",
        option_a="Kafka", option_b="RabbitMQ",
        constraints={"team_size": "medium", "traffic": "high", "experience": "mixed", "budget": "high"},
    ),
    dict(
        question="Should we build the new product as microservices or a monolith?",
        option_a="Microservices", option_b="Monolith",
        constraints={"team_size": "small", "traffic": "low", "experience": "junior", "budget": "low"},
    ),
    dict(
        question="Should we expose our API as REST or GraphQL?",
        option_a="REST", option_b="GraphQL",
        constraints={"team_size": "medium", "traffic": "medium", "experience": "mixed", "budget": "medium"},
    ),
    dict(
        question="Should we use Kubernetes or AWS ECS for container orchestration?",
        option_a="Kubernetes (EKS)", option_b="AWS ECS",
        constraints={"team_size": "small", "traffic": "medium", "experience": "mixed", "budget": "medium"},
    ),
    dict(
        question="Should we use a SQL or NoSQL database for our analytics pipeline?",
        option_a="BigQuery (SQL)", option_b="MongoDB Atlas",
        constraints={"team_size": "medium", "traffic": "low", "experience": "senior", "budget": "high"},
    ),
    dict(
        question="Should we use a monorepo or polyrepo strategy for our codebase?",
        option_a="Monorepo (Turborepo)", option_b="Polyrepo",
        constraints={"team_size": "large", "traffic": "medium", "experience": "mixed", "budget": "medium"},
    ),
    dict(
        question="Should we use server-side rendering or a SPA for our web frontend?",
        option_a="Next.js SSR", option_b="React SPA",
        constraints={"team_size": "small", "traffic": "medium", "experience": "junior", "budget": "low"},
    ),
    dict(
        question="Should we use a managed search service or self-hosted Elasticsearch?",
        option_a="Algolia (managed)", option_b="Elasticsearch (self-hosted)",
        constraints={"team_size": "small", "traffic": "medium", "experience": "mixed", "budget": "low"},
    ),
]


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def token_cost(records: List[UsageRecord]) -> float:
    total = 0.0
    for r in records:
        inp_price, out_price = PRICING.get(r.model, (0.0, 0.0))
        total += (r.input_tokens  / 1_000_000) * inp_price
        total += (r.output_tokens / 1_000_000) * out_price
    return total


def naive_cost(records: List[UsageRecord]) -> float:
    inp_price, out_price = PRICING[NAIVE_MODEL]
    total = 0.0
    for r in records:
        total += (r.input_tokens  / 1_000_000) * inp_price
        total += (r.output_tokens / 1_000_000) * out_price
    return total


def total_tokens(records: List[UsageRecord]) -> tuple[int, int]:
    return (
        sum(r.input_tokens  for r in records),
        sum(r.output_tokens for r in records),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    question: str
    usage:            List[UsageRecord] = field(default_factory=list)
    actual_cost_usd:  float = 0.0
    naive_cost_usd:   float = 0.0
    critic_rejected:  bool  = False


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

    reset_usage()
    final: dict = {}

    async for chunk in parallel_graph.astream(state, stream_mode="updates"):
        for node_name, node_update in chunk.items():
            if isinstance(node_update, dict) and node_name in node_update:
                final[node_name] = node_update[node_name]

    usage   = pop_usage()
    actual  = token_cost(usage)
    naive   = naive_cost(usage)
    critic  = final.get("critic")
    rejected = bool(critic and getattr(critic, "requires_revision", False))

    return RunResult(
        question=d["question"][:60],
        usage=usage,
        actual_cost_usd=actual,
        naive_cost_usd=naive,
        critic_rejected=rejected,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: List[RunResult]) -> None:
    W = 76

    total_actual = sum(r.actual_cost_usd for r in results)
    total_naive  = sum(r.naive_cost_usd  for r in results)
    n_rejected   = sum(1 for r in results if r.critic_rejected)
    n            = len(results)
    mean_actual  = total_actual / n
    mean_naive   = total_naive  / n
    savings_x    = mean_naive / mean_actual if mean_actual else 0
    rejection_pct = (n_rejected / n) * 100

    all_usage = [u for r in results for u in r.usage]
    in_tok, out_tok = total_tokens(all_usage)

    # Per-model breakdown
    model_stats: dict = {}
    for u in all_usage:
        s = model_stats.setdefault(u.model, {"in": 0, "out": 0, "calls": 0, "cost": 0.0})
        s["in"]    += u.input_tokens
        s["out"]   += u.output_tokens
        s["calls"] += 1
        inp_p, out_p = PRICING.get(u.model, (0, 0))
        s["cost"]  += (u.input_tokens / 1e6) * inp_p + (u.output_tokens / 1e6) * out_p

    print()
    print("=" * W)
    print("  DECISIONDESK — COST ACCOUNTING REPORT")
    print(f"  {n} decisions  ·  mixed routing  ·  naive baseline = {NAIVE_MODEL}-everywhere")
    print("=" * W)

    print(f"\n  {'Decision':<62}  {'Actual':>7}  {'Naive':>7}  {'Rej?':>5}")
    print("  " + "─" * (W - 2))
    for r in results:
        rej = "✗" if r.critic_rejected else "✓"
        print(f"  {r.question:<62}  ${r.actual_cost_usd:>5.4f}  ${r.naive_cost_usd:>5.4f}  {rej:>5}")

    print("  " + "═" * (W - 2))
    print(f"  {'MEAN':<62}  ${mean_actual:>5.4f}  ${mean_naive:>5.4f}")
    print()

    print("  Per-model token breakdown")
    print("  " + "─" * (W - 2))
    print(f"  {'Model':<16}  {'Calls':>6}  {'Input tok':>10}  {'Output tok':>11}  {'Cost':>9}")
    for model, s in sorted(model_stats.items()):
        print(f"  {model:<16}  {s['calls']:>6}  {s['in']:>10,}  {s['out']:>11,}  ${s['cost']:>8.4f}")
    print(f"\n  Total tokens: {in_tok:,} input  +  {out_tok:,} output")

    print()
    print("=" * W)
    print(f"  Mean cost per review  (mixed routing)   ${mean_actual:.4f}")
    print(f"  Mean cost per review  ({NAIVE_MODEL} everywhere)  ${mean_naive:.4f}")
    print(f"  Cost savings                             {savings_x:.1f}×  cheaper")
    print("=" * W)
    print()
    print("  Resume bullet fill-in:")
    print()
    print(f'  "Instrumented token-level cost accounting across all 8 agents; routed')
    print(f'   GPT-5 to specialists and GPT-5-mini to the planner/critic, cutting cost')
    print(f'   to under $0.10 per review cycle — {savings_x:.1f}× cheaper than the naive')
    print(f'   single-model approach."')
    print()
    print("=" * W)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(runs: int) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    decisions = (DECISIONS * ((runs // len(DECISIONS)) + 1))[:runs]

    print(f"\n  Running {runs} decisions ({runs * 8} LLM calls)...\n")

    results: List[RunResult] = []
    for i, d in enumerate(decisions, 1):
        label = d["question"][:55]
        print(f"  [{i:>2}/{runs}] {label}...", end=" ", flush=True)
        r = await run_decision(d)
        results.append(r)
        rej = "REJECTED" if r.critic_rejected else "approved"
        print(f"${r.actual_cost_usd:.4f}  critic:{rej}")

    print_report(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of decisions to run (default 10)")
    args = parser.parse_args()
    asyncio.run(main(args.runs))

"""DecisionDesk — FastAPI backend.

Endpoints:
  GET  /            → serves index.html
  GET  /health      → {"status": "ok"}
  GET  /metrics     → Prometheus metrics (Phase 3)
  POST /review      → one-shot review, returns full agent trace
  WS   /ws/review   → real-time streaming review, emits per-agent events
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from decisiondesk.core.schema import (
    PlannerOutput, SpecialistOutput, DetectorOutput,
)
from decisiondesk.core.state import GateTier
from decisiondesk.graph.nodes import LGState
from decisiondesk.graph.parallel import parallel_graph
from decisiondesk.graph.revision import revision_graph
from decisiondesk.graph.sequential import sequential_graph

# index.html lives at the project root (two levels above this file's src/decisiondesk/)
INDEX_HTML = Path(__file__).resolve().parents[2] / "index.html"

app = FastAPI(title="DecisionDesk", version="0.1.0")

# All LangGraph node names — used to filter astream_events noise
AGENT_NODES = frozenset({
    "planner", "scalability", "security", "cost",
    "maintainability", "detector", "critic", "synthesizer", "gate",
})

# Model assigned to each agent — surfaced in the /review response for cost accounting
AGENT_MODELS: Dict[str, Optional[str]] = {
    "planner":         "gpt-5-mini",
    "scalability":     "gpt-5",
    "security":        "gpt-5",
    "cost":            "gpt-5",
    "maintainability": "gpt-5",
    "detector":        "gpt-5-mini",
    "critic":          "gpt-5-mini",
    "synthesizer":     "gpt-5.5",
    "gate":            None,          # rule-based, no LLM
}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ReviewRequest(BaseModel):
    question:      str
    option_a:      str
    option_b:      str
    constraints:   Dict[str, str] = {}
    mode:          str  = "parallel"   # "parallel" | "sequential"
    force_approve: bool = False


class AgentTrace(BaseModel):
    agent:  str
    model:  Optional[str]
    output: Optional[Dict[str, Any]]


class ReviewResponse(BaseModel):
    mode:               str
    approved:           bool
    gate_tier:          str
    gate_reason:        str
    recommendation:     Optional[str]
    confidence:         Optional[float]
    rationale:          Optional[str]
    tradeoffs:          List[str] = []
    unresolved_risks:   List[str] = []
    agents:             List[AgentTrace] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_lg_state(req: ReviewRequest) -> LGState:
    return LGState(
        question=req.question,
        option_a=req.option_a,
        option_b=req.option_b,
        constraints=req.constraints,
        iteration=1,
        gate_tier=GateTier.EXPLORATION.value,
        force_approve=req.force_approve,
    )


def _serialise(value: Any) -> Any:
    """Recursively convert Pydantic models to plain dicts."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _build_response(mode: str, result: Dict[str, Any]) -> ReviewResponse:
    gate        = result.get("gate")
    synthesizer = result.get("synthesizer")

    agents = [
        AgentTrace(
            agent=name,
            model=AGENT_MODELS.get(name),
            output=_serialise(result.get(name)),
        )
        for name in [
            "planner", "scalability", "security", "cost",
            "maintainability", "detector", "critic", "synthesizer", "gate",
        ]
    ]

    return ReviewResponse(
        mode=mode,
        approved=gate.decision.approved if gate else False,
        gate_tier=gate.decision.tier if gate else GateTier.EXPLORATION.value,
        gate_reason=gate.decision.reason if gate else "",
        recommendation=synthesizer.final_recommendation if synthesizer else None,
        confidence=synthesizer.confidence if synthesizer else None,
        rationale=synthesizer.rationale if synthesizer else None,
        tradeoffs=synthesizer.tradeoffs if synthesizer else [],
        unresolved_risks=synthesizer.unresolved_risks if synthesizer else [],
        agents=agents,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    # Phase 3: replace with prometheus_client.generate_latest()
    return {"note": "Prometheus metrics endpoint — wired in Phase 3"}


@app.post("/review", response_model=ReviewResponse)
def run_review(req: ReviewRequest):
    """
    One-shot review. Runs the full pipeline synchronously and returns
    the complete agent trace. Use the WebSocket endpoint for real-time
    streaming.
    """
    graph = parallel_graph if req.mode == "parallel" else sequential_graph
    try:
        result = graph.invoke(_build_lg_state(req))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return _build_response(req.mode, result)


@app.websocket("/ws/review")
async def review_websocket(ws: WebSocket):
    """
    Real-time streaming review.

    Client sends a single JSON message to start:
      {"type": "start", "question": "...", "option_a": "...", "option_b": "...",
       "constraints": {}, "force_approve": false}

    Server emits:
      {"type": "state_initialized", "timestamp": ...}
      {"type": "agent_started",     "agent": "planner", "timestamp": ...}
      {"type": "agent_completed",   "agent": "planner", "output": {...}, "timestamp": ...}
      ...  (one pair per agent; parallel specialists arrive as they finish)
      {"type": "review_complete",   "approved": bool, "recommendation": "...", ...}
      {"type": "error",             "error": "...", "timestamp": ...}
    """
    await ws.accept()

    async def emit(event: dict) -> None:
        await ws.send_json(event)
        await asyncio.sleep(0)

    try:
        msg = await ws.receive_json()
        if msg.get("type") != "start":
            await ws.close(code=1003, reason="Expected 'start' message")
            return

        state = LGState(
            question=msg["question"],
            option_a=msg["option_a"],
            option_b=msg["option_b"],
            constraints=msg.get("constraints", {}),
            iteration=1,
            gate_tier=GateTier.EXPLORATION.value,
            force_approve=msg.get("force_approve", False),
        )

        await emit({"type": "state_initialized", "timestamp": time.time()})

        # Accumulate outputs keyed by agent name for the final event
        final_state: Dict[str, Any] = {}

        # astream(stream_mode="updates") yields one chunk per completed node
        # (or one chunk with all parallel nodes when a super-step finishes).
        # Each chunk is {node_name: what_the_node_returned}.
        async for chunk in parallel_graph.astream(state, stream_mode="updates"):
            for node_name, node_update in chunk.items():
                if node_name not in AGENT_NODES:
                    continue

                # node_update is the dict the node function returned, e.g. {"planner": PlannerOutput}
                agent_output = node_update.get(node_name) if isinstance(node_update, dict) else None
                final_state[node_name] = agent_output

                await emit({
                    "type":      "agent_started",
                    "agent":     node_name,
                    "model":     AGENT_MODELS.get(node_name),
                    "timestamp": time.time(),
                })
                await emit({
                    "type":      "agent_completed",
                    "agent":     node_name,
                    "model":     AGENT_MODELS.get(node_name),
                    "output":    _serialise(agent_output),
                    "timestamp": time.time(),
                })

        gate        = final_state.get("gate")
        synthesizer = final_state.get("synthesizer")

        await emit({
            "type":             "review_complete",
            "approved":         gate.decision.approved if gate else False,
            "gate_tier":        gate.decision.tier if gate else GateTier.EXPLORATION.value,
            "gate_reason":      gate.decision.reason if gate else "",
            "recommendation":   synthesizer.final_recommendation if synthesizer else None,
            "confidence":       synthesizer.confidence if synthesizer else None,
            "rationale":        synthesizer.rationale if synthesizer else None,
            "tradeoffs":        synthesizer.tradeoffs if synthesizer else [],
            "unresolved_risks": synthesizer.unresolved_risks if synthesizer else [],
            "timestamp":        time.time(),
        })

    except WebSocketDisconnect:
        pass

    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            await emit({
                "type":      "error",
                "error":     str(exc),
                "timestamp": time.time(),
            })
        except Exception:
            pass


@app.websocket("/ws/revise")
async def revise_websocket(ws: WebSocket):
    """
    Human-in-the-loop revision pass.

    Client sends one JSON message:
      {
        "type": "revise",
        "question": "...", "option_a": "...", "option_b": "...", "constraints": {},
        "accepted_issues":    ["issue text the user acknowledged", ...],
        "additional_context": "free-form user notes",
        "force_approve": false,
        "planner":         { ...PlannerOutput... },
        "scalability":     { ...SpecialistOutput... },
        "security":        { ...SpecialistOutput... },
        "cost":            { ...SpecialistOutput... },
        "maintainability": { ...SpecialistOutput... },
        "detector":        { ...DetectorOutput... }
      }

    Server re-runs synthesizer → gate (critic is skipped; user reviewing issues IS the
    critic pass). The critic output is pre-populated as cleared, force_approve=True.
    """
    await ws.accept()

    async def emit(event: dict) -> None:
        await ws.send_json(event)
        await asyncio.sleep(0)

    try:
        msg = await ws.receive_json()
        if msg.get("type") != "revise":
            await ws.close(code=1003, reason="Expected 'revise' message")
            return

        # Build revision_notes from user selections
        from decisiondesk.core.schema import CriticOutput  # local import avoids circular at module level
        accepted = msg.get("accepted_issues", [])
        extra    = (msg.get("additional_context") or "").strip()
        parts: List[str] = []
        if accepted:
            parts.append("Issues user acknowledged:\n" + "\n".join(f"- {i}" for i in accepted))
        if extra:
            parts.append("Additional context from user:\n" + extra)
        revision_notes = "\n\n".join(parts) or None

        # Reconstruct previous agent outputs from the JSON the client sent back
        def _parse(cls, key):
            raw = msg.get(key)
            return cls.model_validate(raw) if raw else None

        # User reviewing issues = critic cleared. Pre-populate an empty critic output
        # so the synthesizer can run without a critic node in the revision graph.
        cleared_critic = CriticOutput(
            issues=[],
            requires_revision=False,
            revised_constraints=[],
        )

        state = LGState(
            question=msg["question"],
            option_a=msg["option_a"],
            option_b=msg["option_b"],
            constraints=msg.get("constraints", {}),
            iteration=1,
            gate_tier=GateTier.EXPLORATION.value,
            force_approve=True,          # user chose to proceed — gate always approves
            revision_notes=revision_notes,
            # Pre-populate all previous outputs
            planner=_parse(PlannerOutput, "planner"),
            scalability=_parse(SpecialistOutput, "scalability"),
            security=_parse(SpecialistOutput, "security"),
            cost=_parse(SpecialistOutput, "cost"),
            maintainability=_parse(SpecialistOutput, "maintainability"),
            detector=_parse(DetectorOutput, "detector"),
            critic=cleared_critic,
        )

        await emit({"type": "state_initialized", "timestamp": time.time()})

        final_state: Dict[str, Any] = {}

        async for chunk in revision_graph.astream(state, stream_mode="updates"):
            for node_name, node_update in chunk.items():
                if node_name not in AGENT_NODES:
                    continue
                agent_output = node_update.get(node_name) if isinstance(node_update, dict) else None
                final_state[node_name] = agent_output
                await emit({
                    "type":      "agent_started",
                    "agent":     node_name,
                    "model":     AGENT_MODELS.get(node_name),
                    "timestamp": time.time(),
                })
                await emit({
                    "type":      "agent_completed",
                    "agent":     node_name,
                    "model":     AGENT_MODELS.get(node_name),
                    "output":    _serialise(agent_output),
                    "timestamp": time.time(),
                })

        gate        = final_state.get("gate")
        synthesizer = final_state.get("synthesizer")

        await emit({
            "type":             "review_complete",
            "approved":         gate.decision.approved if gate else False,
            "gate_tier":        gate.decision.tier if gate else GateTier.EXPLORATION.value,
            "gate_reason":      gate.decision.reason if gate else "",
            "recommendation":   synthesizer.final_recommendation if synthesizer else None,
            "confidence":       synthesizer.confidence if synthesizer else None,
            "rationale":        synthesizer.rationale if synthesizer else None,
            "tradeoffs":        synthesizer.tradeoffs if synthesizer else [],
            "unresolved_risks": synthesizer.unresolved_risks if synthesizer else [],
            "timestamp":        time.time(),
        })

    except WebSocketDisconnect:
        pass

    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            await emit({
                "type":      "error",
                "error":     str(exc),
                "timestamp": time.time(),
            })
        except Exception:
            pass

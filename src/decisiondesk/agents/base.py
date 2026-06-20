import threading
from dataclasses import dataclass
from typing import Any, List, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Per-call usage tracking (thread-safe — specialists run concurrently)
# ---------------------------------------------------------------------------

@dataclass
class UsageRecord:
    model: str
    input_tokens: int
    output_tokens: int

_lock = threading.Lock()
_session_usage: List[UsageRecord] = []


def reset_usage() -> None:
    with _lock:
        _session_usage.clear()


def pop_usage() -> List[UsageRecord]:
    """Return and clear all accumulated usage records."""
    with _lock:
        records = list(_session_usage)
        _session_usage.clear()
        return records


# ---------------------------------------------------------------------------
# Structured completion
# ---------------------------------------------------------------------------

def complete_structured(
    prompt: str,
    model: str,
    output_schema: Type[T],
    system: str = "You are a senior software architect. Respond only with valid JSON.",
) -> T:
    client = _get_client()
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        text_format=output_schema,
    )

    # Capture token usage — Responses API uses input_tokens / output_tokens.
    # output_tokens includes reasoning tokens (hidden CoT) for thinking models —
    # strip those so we only bill for visible output tokens.
    usage = getattr(response, "usage", None)
    if usage is not None:
        in_tok  = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0

        # Subtract reasoning tokens if the model reported them
        out_details = getattr(usage, "output_tokens_details", None)
        if out_details is not None:
            reasoning = getattr(out_details, "reasoning_tokens", 0) or 0
            out_tok   = max(0, out_tok - reasoning)

        # input_tokens_details.cached_tokens are billed at ~10% of input rate;
        # keep them in input_tokens — cost_report prices them at full rate, so
        # actual spend is slightly lower than reported (conservative estimate).
        with _lock:
            _session_usage.append(UsageRecord(
                model=model,
                input_tokens=in_tok,
                output_tokens=out_tok,
            ))

    return response.output_parsed


def dump(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value

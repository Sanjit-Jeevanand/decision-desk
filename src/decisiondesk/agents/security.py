"""Security specialist — evaluates attack surface, auth boundaries, and compliance posture."""

from typing import Dict

from decisiondesk.agents.base import complete_structured
from decisiondesk.core.schema import SpecialistOutput
from decisiondesk.core.state import DecisionState

MODEL = "gpt-5"

_SYSTEM = (
    "You are a senior application security engineer who specialises in backend "
    "infrastructure decisions. You evaluate technology choices through one lens only: "
    "security posture — attack surface, authentication and authorisation boundaries, "
    "encryption guarantees, compliance obligations, and vulnerability exposure. "
    "You do not consider cost, performance, or team dynamics. Be direct and specific. "
    "Name concrete CVEs, compliance frameworks, or known vulnerabilities where relevant. "
    "Respond only with valid JSON."
)

_PROMPT = """\
You are the Security Specialist in an architecture review panel.

DECISION
--------
Question : {question}
Option A : {option_a}
Option B : {option_b}

PLANNER CONTEXT
---------------
Framing      : {decision_framing}
Your question: {security_question}

CONSTRAINTS (all reviewers must respect these)
-----------------------------------------------
{key_constraints}

IMPORTANT: You cannot see what other specialists think. Reason independently.

YOUR ANALYSIS MUST COVER:

1. ATTACK SURFACE
   - What does each option expose to the network, to the application layer,
     and to internal services?
   - Which option gives you a smaller, more auditable attack surface?
   - Does one option require opening ports, granting broader IAM permissions,
     or running additional daemons that the other does not?

2. AUTHENTICATION & AUTHORISATION
   - How does each option handle access control at the data layer?
   - Does one offer finer-grained access control (row-level security, IAM roles,
     attribute-based access) that is relevant to this decision?

3. ENCRYPTION
   - Encryption at rest: which option provides it by default vs requires configuration?
   - Encryption in transit: TLS version, certificate management, mutual TLS support?

4. COMPLIANCE & DATA RESIDENCY
   - If the constraints mention or imply a compliance framework (GDPR, HIPAA, SOC2,
     PCI-DSS), which option makes it easier to satisfy?
   - Data residency: does one option offer better control over where data is stored?

5. VULNERABILITY EXPOSURE & PATCHING
   - Who is responsible for patching each option?
   - Is there a meaningful difference in historical CVE exposure or patch latency?

RECOMMENDATION RULES:
- Choose "option_a" or "option_b" if one is clearly more secure for this context.
- Choose "neutral" only if security posture is genuinely equivalent — justify it.
- Risks: name specific threat vectors, not categories like "could be hacked".
- Assumptions: list what you assumed about the environment that was not in the constraints.

Respond with JSON matching this schema exactly:
{{
  "recommendation": "option_a" | "option_b" | "neutral",
  "confidence": <float 0.0–1.0>,
  "rationale": "<3–5 sentence explanation grounded in the two specific options>",
  "risks": ["<specific risk>", ...],
  "assumptions": ["<assumption made>", ...]
}}
"""


def run_security(state: DecisionState) -> Dict:
    planner = state.current.planner
    if planner is None:
        raise RuntimeError("Planner must run before security specialist")

    constraints_text = "\n".join(f"- {c}" for c in planner.key_constraints) or "None specified."

    prompt = _PROMPT.format(
        question=state.input.question,
        option_a=state.input.option_a,
        option_b=state.input.option_b,
        decision_framing=planner.decision_framing,
        security_question=planner.security_question,
        key_constraints=constraints_text,
    )

    output = complete_structured(
        prompt=prompt,
        model=MODEL,
        output_schema=SpecialistOutput,
        system=_SYSTEM,
    )

    return {"security": output}

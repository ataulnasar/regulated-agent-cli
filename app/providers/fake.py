"""Deterministic, offline stand-in for Claude.

Two modes:
- Scripted: replays a fixed list of turns. Evals use this to simulate specific behaviour,
  including a misbehaving or compromised model: a hallucinated product, an invented citation,
  or {"approved": true}.
- Heuristic (default): a small rule-based agent that uses the tools the way a well-behaved model
  would, so the CLI demo works without an API key.

Fake-provider runs test the harness and the guardrails, not model quality. Real-model evals do that.
"""

import json
from typing import Any

from app.prompts import parse_task, unwrap_tool_output
from app.providers import Block, Message, ModelResponse

# A scripted turn is a list of steps: {"text": "..."} or {"tool": "<name>", "input": {...}}.
Turn = list[dict[str, Any]]

INTENT_KEYWORDS = [
    ("retirement_planning", ("retire", "retirement", "pension")),
    ("growth_investing", ("growth", "grow", "equity", "equities", "return")),
    ("savings", ("save", "saving", "savings")),
]


class FakeProvider:
    name = "fake"

    def __init__(self, script: list[Turn] | None = None, model_id: str = "fake-heuristic") -> None:
        self.model_id = model_id if script is None else "fake-scripted"
        self._script = script
        self._turn = 0
        # Every request the "model" received, so tests can check exactly what was sent to it.
        self.requests: list[dict[str, Any]] = []

    def create(
        self, system: str, messages: list[Message], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        self.requests.append({"system": system, "messages": list(messages), "tools": tools})
        steps = self._scripted_turn() if self._script is not None else _heuristic_turn(messages)
        self._turn += 1
        if steps is None:
            return ModelResponse(
                [{"type": "text", "text": "I have nothing more to add."}], "end_turn"
            )
        content = [self._block(step, index) for index, step in enumerate(steps)]
        stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
        return ModelResponse(content, stop)

    def _scripted_turn(self) -> Turn | None:
        assert self._script is not None
        return self._script[self._turn] if self._turn < len(self._script) else None

    def _block(self, step: dict[str, Any], index: int) -> Block:
        if "tool" in step:
            return {
                "type": "tool_use",
                "id": f"toolu_fake_{self._turn}_{index}",
                "name": step["tool"],
                "input": step.get("input", {}),
            }
        return {"type": "text", "text": step["text"]}


# --- Heuristic agent ---------------------------------------------------------------------------


def _heuristic_turn(messages: list[Message]) -> Turn | None:
    customer_id, request = parse_task(messages[0]["content"])
    results = _tool_results(messages)
    intent = _intent(request)

    if "get_customer_profile" not in results:
        return [
            {"text": "Let me look up the customer's profile."},
            {"tool": "get_customer_profile", "input": {"customer_id": customer_id}},
        ]
    if "search_product_catalog" not in results:
        search_query = {"retirement_planning": "low risk retirement"}.get(intent, request)
        return [
            {"tool": "search_product_catalog", "input": {"query": search_query}},
            {"tool": "retrieve_policy_context", "input": {"query": request}},
        ]

    products = results["search_product_catalog"][0] or []
    if products and "check_product_eligibility" not in results:
        return [
            {
                "tool": "check_product_eligibility",
                "input": {"customer_id": customer_id, "product_id": p["product_id"]},
            }
            for p in products
        ]
    if "submit_recommendation" in results:
        return None
    return [{"tool": "submit_recommendation", "input": _recommendation(intent, results)}]


def _recommendation(intent: str, results: dict[str, list[Any]]) -> dict[str, Any]:
    products = results["search_product_catalog"][0] or []
    eligibility = results.get("check_product_eligibility", [])
    eligible = [p for p, e in zip(products, eligibility, strict=False) if e and e["allowed"]]
    policy_citations = [
        s["citation"] for s in results.get("retrieve_policy_context", [[]])[0] or []
    ]

    if intent == "unknown":
        return {
            "intent": intent,
            "recommended_action": "advisor_review",
            "reasoning_summary": "No clear goal in the request; an advisor should clarify.",
            "citations": policy_citations,
        }
    if not eligible:
        return {
            "intent": intent,
            "recommended_action": "need_more_info",
            "reasoning_summary": "No product in the catalogue passed the eligibility check.",
            "citations": policy_citations,
        }
    choice = eligible[0]
    return {
        "intent": intent,
        "recommended_action": "recommend_product",
        "recommended_product_id": choice["product_id"],
        "reasoning_summary": f"{choice['name']} is the best-matching product that passed the "
        "eligibility check for this customer's risk tolerance and horizon.",
        "citations": [*policy_citations, choice["source_id"]],
        "model_confidence": 0.8,
    }


def _intent(request: str) -> str:
    words = request.lower()
    for intent, keywords in INTENT_KEYWORDS:
        if any(k in words for k in keywords):
            return intent
    return "unknown"


def _tool_results(messages: list[Message]) -> dict[str, list[Any]]:
    """Tool name -> parsed results, in call order. Failed calls are recorded as None."""
    names: dict[str, str] = {}
    results: dict[str, list[Any]] = {}
    for message in messages:
        if not isinstance(message["content"], list):
            continue
        for block in message["content"]:
            if block.get("type") == "tool_use":
                names[block["id"]] = block["name"]
            elif block.get("type") == "tool_result":
                name = names.get(block["tool_use_id"], "unknown")
                parsed = None if block.get("is_error") else _parse(block.get("content"))
                results.setdefault(name, []).append(parsed)
    return results


def _parse(content: Any) -> Any:
    try:
        return json.loads(unwrap_tool_output(content)) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return None

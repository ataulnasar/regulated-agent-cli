"""The agent loop: one model, an allowlist of tools, a bounded number of steps.

The model decides which tools to call and in what order, and finishes by calling
submit_recommendation with a schema-validated AgentRecommendation. The harness around it:
- only executes allowlisted tools, through the validated registry;
- records which citations were actually retrieved, from real tool output, never from the model;
- validates the submission, allows one corrected retry, and otherwise falls back to human review;
- hands the result to the deterministic guardrail, which sets the status. The model never does.

Why one agent and not several: the task is a single, short, auditable decision. More agents would
add hand-offs, latency and places for context to get lost, without adding a capability. The
natural split, if it grows, is an independent compliance-review agent with its own separate
context that critiques the draft. The deterministic guardrail would still have the final say.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from pydantic import ValidationError
from pydantic_core import to_json

from app.audit import append_event
from app.data import load_products
from app.guardrails import evaluate, status_for
from app.models import (
    AgentRecommendation,
    AssessmentMetrics,
    CustomerProfile,
    FinalAssessment,
    Intent,
    PolicySection,
    Product,
    RecommendedAction,
    ToolCall,
)
from app.observability import fingerprint, log_event, trace_id_var
from app.prompts import SYSTEM_PROMPT, format_task
from app.providers import Block, Message, ModelProvider, ModelResponse
from app.tools import TOOLS, execute_tool, load_customer_profile, tool_schema

logger = logging.getLogger(__name__)

MAX_STEPS = 8
MAX_CORRECTIONS = 1
SUBMIT_TOOL = "submit_recommendation"
SUBMIT_TOOL_DEFINITION = {
    "name": SUBMIT_TOOL,
    "description": "Submit your final recommendation. Call exactly once, as your last action.",
    "input_schema": tool_schema(AgentRecommendation),
}


@dataclass
class _Run:
    """Mutable state of one assessment run."""

    messages: list[Message]
    tool_calls: list[ToolCall] = field(default_factory=list)
    retrieved_citations: set[str] = field(default_factory=set)
    recommendation: AgentRecommendation | None = None
    corrections_left: int = MAX_CORRECTIONS
    failure: str | None = None
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def tool_definitions() -> list[dict[str, Any]]:
    return [spec.definition() for spec in TOOLS.values()] + [SUBMIT_TOOL_DEFINITION]


def run_assessment(
    customer_id: str, advisor_id: str, request: str, provider: ModelProvider
) -> FinalAssessment:
    """Run one assessment end to end and write it to the audit trail."""
    # Fail fast on an unknown customer, before spending a model call.
    profile = load_customer_profile(customer_id)
    trace_id = uuid4().hex[:12]
    token = trace_id_var.set(trace_id)
    try:
        return _run(profile, advisor_id, request, provider, trace_id)
    finally:
        trace_id_var.reset(token)


def _run(
    profile: CustomerProfile,
    advisor_id: str,
    request: str,
    provider: ModelProvider,
    trace_id: str,
) -> FinalAssessment:
    started = time.perf_counter()
    log_event(
        logger,
        "assessment_started",
        customer_id=profile.customer_id,
        advisor_id=advisor_id,
        provider=provider.name,
        model_id=provider.model_id,
        # Never the request text itself: it may contain personal data.
        request_chars=len(request),
        request_sha256=fingerprint(request),
    )
    run = _Run(messages=[{"role": "user", "content": format_task(profile.customer_id, request)}])
    tools = tool_definitions()

    for step in range(1, MAX_STEPS + 1):
        response = _call_model(run, provider, tools, step)
        run.messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            _handle_stop_without_tools(run, response.stop_reason)
        else:
            results = [_handle_tool_use(run, block) for block in response.tool_uses()]
            run.messages.append({"role": "user", "content": results})
        if run.recommendation or run.failure:
            break
    else:
        run.failure = f"no valid recommendation within {MAX_STEPS} steps"

    recommendation = run.recommendation or _fallback(run.failure or "unknown failure")
    compliance = evaluate(
        profile,
        recommendation,
        run.retrieved_citations,
        load_products(),
        model_output_invalid=run.recommendation is None,
    )
    metrics = AssessmentMetrics(
        model_calls=run.model_calls,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    assessment = FinalAssessment(
        trace_id=trace_id,
        customer_id=profile.customer_id,
        advisor_id=advisor_id,
        model=f"{provider.name}/{provider.model_id}",
        request=request,
        recommendation=recommendation,
        compliance=compliance,
        tool_calls=run.tool_calls,
        metrics=metrics,
        status=status_for(recommendation, compliance),
    )
    log_event(
        logger,
        "assessment_completed",
        assessment_id=str(assessment.assessment_id),
        status=assessment.status,
        allowed=compliance.allowed,
        requires_human_review=compliance.requires_human_review,
        rules_triggered=compliance.rules_triggered,
        citations=recommendation.citations,
        fallback_reason=run.failure,
        **metrics.model_dump(),
    )
    append_event("assessment_created", assessment, actor="system")
    return assessment


def _call_model(
    run: _Run, provider: ModelProvider, tools: list[dict[str, Any]], step: int
) -> ModelResponse:
    started = time.perf_counter()
    response = provider.create(SYSTEM_PROMPT, run.messages, tools)
    run.model_calls += 1
    run.input_tokens += response.input_tokens
    run.output_tokens += response.output_tokens
    log_event(
        logger,
        "model_call",
        step=step,
        provider=provider.name,
        model_id=provider.model_id,
        stop_reason=response.stop_reason,
        tool_uses=[b["name"] for b in response.tool_uses()],
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return response


def _handle_tool_use(run: _Run, block: Block) -> Block:
    """Execute one tool_use block and return its tool_result."""
    if block["name"] == SUBMIT_TOOL:
        return _handle_submission(run, block)

    result, call = execute_tool(block["name"], block["input"])
    run.tool_calls.append(call)
    if not call.success:
        return _tool_result(block, call.result_summary, is_error=True)
    run.retrieved_citations |= _citations_in(result)
    return _tool_result(block, to_json(result).decode())


def _handle_submission(run: _Run, block: Block) -> Block:
    try:
        recommendation = AgentRecommendation.model_validate(block["input"])
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors()
        )
        run.tool_calls.append(_submit_call(block, success=False, summary=f"invalid: {errors}"))
        if run.corrections_left > 0:
            run.corrections_left -= 1
            return _tool_result(block, f"Invalid recommendation: {errors}. Fix it.", is_error=True)
        run.failure = "recommendation failed validation twice"
        return _tool_result(block, "Invalid recommendation.", is_error=True)

    run.recommendation = recommendation
    product = recommendation.recommended_product_id
    summary = f"accepted: {recommendation.recommended_action}" + (f" {product}" if product else "")
    run.tool_calls.append(_submit_call(block, success=True, summary=summary))
    return _tool_result(block, "Recommendation received.")


def _handle_stop_without_tools(run: _Run, stop_reason: str) -> None:
    """The model stopped without submitting. Nudge once, then give up safely."""
    if stop_reason == "end_turn" and run.corrections_left > 0:
        run.corrections_left -= 1
        run.messages.append({"role": "user", "content": f"Finish by calling {SUBMIT_TOOL}."})
        return
    run.failure = f"model stopped without a recommendation (stop_reason={stop_reason})"


def _citations_in(result: Any) -> set[str]:
    """Citations the model has legitimately seen: retrieved policy sections and found products."""
    items = result if isinstance(result, list) else [result]
    citations = set()
    for item in items:
        if isinstance(item, PolicySection):
            citations.add(item.citation)
        elif isinstance(item, Product):
            citations.add(item.source_id)
    return citations


def _fallback(reason: str) -> AgentRecommendation:
    log_event(logger, "fallback_recommendation", logging.WARNING, reason=reason)
    return AgentRecommendation(
        intent=Intent.UNKNOWN,
        recommended_action=RecommendedAction.ADVISOR_REVIEW,
        reasoning_summary=f"No valid model recommendation ({reason}). Escalated to an advisor.",
    )


def _tool_result(block: Block, content: str, is_error: bool = False) -> Block:
    result: Block = {"type": "tool_result", "tool_use_id": block["id"], "content": content}
    if is_error:
        result["is_error"] = True
    return result


def _submit_call(block: Block, *, success: bool, summary: str) -> ToolCall:
    return ToolCall(
        tool_name=SUBMIT_TOOL,
        arguments=block["input"] if isinstance(block["input"], dict) else {},
        success=success,
        result_summary=summary[:300],
        duration_ms=0,
    )

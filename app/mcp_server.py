"""Optional MCP server: the same capabilities, for any MCP client (Claude Code, Claude Desktop).

    uv run --extra mcp python -m app.mcp_server

Nothing is reimplemented here:
- the four business tools go through the same registry and execute_tool() as the agent, so inputs
  are validated by the same Pydantic models and every call is logged the same way;
- assess_request runs the full run_assessment(), guardrail and audit trail included.

There is deliberately no approve/review tool. An MCP client is software acting for someone; approval
stays with an identified second advisor via the CLI (four-eyes). Assessments that need review are
put in the pending queue for that.
"""

import asyncio
from typing import Annotated, Any

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from pydantic import StringConstraints, ValidationError
from pydantic_core import to_json

from app.agent import run_assessment
from app.models import AdvisorId, AssessmentStatus, CustomerId, FinalAssessment
from app.observability import setup_logging
from app.providers import get_provider
from app.review import save_pending
from app.tools import TOOLS, DomainError, ToolInput, execute_tool, tool_schema

INSTRUCTIONS = (
    "Suitability tools for a fictional Nordic wealth manager. Business tools are read-only. "
    "assess_request runs the full agent with deterministic compliance rules; its status is final "
    "unless it is pending_review, which a second human advisor must decide outside MCP."
)


class AssessRequestInput(ToolInput):
    customer_id: CustomerId
    advisor_id: AdvisorId
    request: Annotated[str, StringConstraints(min_length=1, max_length=2000)]


ASSESS_TOOL = types.Tool(
    name="assess_request",
    description=(
        "Run a complete suitability assessment for a customer request: the agent drafts a "
        "recommendation and deterministic compliance rules decide the outcome. Returns the status, "
        "the recommendation, triggered rules and citations. Cannot approve pending reviews."
    ),
    input_schema=tool_schema(AssessRequestInput),
)


def tool_list() -> list[types.Tool]:
    return [types.Tool(**spec.definition()) for spec in TOOLS.values()] + [ASSESS_TOOL]


def assessment_summary(assessment: FinalAssessment) -> dict[str, Any]:
    rec, compliance = assessment.recommendation, assessment.compliance
    return {
        "assessment_id": str(assessment.assessment_id),
        "trace_id": assessment.trace_id,
        "status": assessment.status,
        "released": assessment.released,
        "intent": rec.intent,
        "recommended_action": rec.recommended_action,
        "recommended_product_id": rec.recommended_product_id,
        "reasoning_summary": rec.reasoning_summary,
        "citations": rec.citations,
        "rules_triggered": compliance.rules_triggered,
        "reasons": compliance.reasons,
    }


def _assess(arguments: dict[str, Any]) -> dict[str, Any]:
    args = AssessRequestInput.model_validate(arguments)
    assessment = run_assessment(args.customer_id, args.advisor_id, args.request, get_provider())
    if assessment.status == AssessmentStatus.PENDING_REVIEW:
        save_pending(assessment)
    return assessment_summary(assessment)


def _result(payload: Any, is_error: bool = False) -> types.CallToolResult:
    text = payload if isinstance(payload, str) else to_json(payload).decode()
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)], is_error=is_error
    )


async def _list_tools(_ctx: Any, _params: Any) -> types.ListToolsResult:
    return types.ListToolsResult(tools=tool_list())


async def _call_tool(_ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
    arguments = params.arguments or {}
    if params.name == ASSESS_TOOL.name:
        try:
            # The agent may call a real model: keep it off the event loop.
            return _result(await asyncio.to_thread(_assess, arguments))
        except (ValidationError, DomainError, ValueError) as exc:
            return _result(f"error: {exc}", is_error=True)

    result, call = execute_tool(params.name, arguments)
    if not call.success:
        return _result(call.result_summary, is_error=True)
    return _result(result)


server: Server[Any] = Server(
    "regulated-agent",
    version="0.1.0",
    instructions=INSTRUCTIONS,
    on_list_tools=_list_tools,
    on_call_tool=_call_tool,
)


async def _serve() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    # Logs go to stderr; stdout belongs to the MCP protocol.
    setup_logging("WARNING")
    asyncio.run(_serve())


if __name__ == "__main__":
    main()

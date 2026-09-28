"""Rich terminal rendering of an assessment, laid out for a live walkthrough.

This is the advisor's view, so it shows the customer's name. The model never sees it.
"""

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.data import load_products
from app.models import AssessmentStatus, FinalAssessment
from app.rag import policy_sections
from app.tools import load_customer_profile

STATUS_STYLE = {
    AssessmentStatus.AUTO_APPROVED: ("green", "✓ AUTO-APPROVED"),
    AssessmentStatus.APPROVED_BY_HUMAN: ("green", "✓ APPROVED BY HUMAN REVIEWER"),
    AssessmentStatus.PENDING_REVIEW: ("yellow", "⚠ HUMAN REVIEW REQUIRED"),
    AssessmentStatus.REJECTED: ("red", "✗ REJECTED"),
    AssessmentStatus.DECLINED: ("blue", "– DECLINED"),
}


def render_assessment(console: Console, assessment: FinalAssessment) -> None:
    rec = assessment.recommendation
    profile = load_customer_profile(assessment.customer_id)

    console.rule(f"[bold]REGULATED AGENT[/]  trace {assessment.trace_id}  ·  {assessment.model}")
    if metrics := assessment.metrics:
        console.print(
            f"[dim]{metrics.model_calls} model calls · {metrics.input_tokens} in / "
            f"{metrics.output_tokens} out tokens · {metrics.latency_ms:.0f} ms[/]",
            justify="center",
        )

    customer = Table.grid(padding=(0, 2))
    customer.add_row("Name", profile.name)
    customer.add_row("Customer ID", profile.customer_id)
    customer.add_row("Risk tolerance", str(profile.risk_tolerance or "MISSING").upper())
    horizon = profile.investment_horizon_years
    customer.add_row("Horizon", f"{horizon} years" if horizon is not None else "MISSING")
    customer.add_row("Requested by", assessment.advisor_id)
    console.print(Panel(customer, title="Customer", title_align="left"))
    console.print(Panel(Text(assessment.request), title="Request", title_align="left"))

    tools = Table(box=None, show_header=False, padding=(0, 1))
    for call in assessment.tool_calls:
        mark = "[green]✓[/]" if call.success else "[red]✗[/]"
        tools.add_row(mark, call.tool_name, f"[dim]{call.result_summary}[/]")
    agent = Group(
        Text.from_markup(f"Intent: [bold]{rec.intent}[/]"),
        Text(f"Reasoning: {rec.reasoning_summary}"),
        Text(""),
        tools,
    )
    console.print(Panel(agent, title="Agent", title_align="left"))

    evidence = Table(box=None, show_header=False, padding=(0, 1))
    for i, citation in enumerate(rec.citations, start=1):
        evidence.add_row(f"[{i}]", citation, f"[dim]{describe_citation(citation)}[/]")
    console.print(
        Panel(evidence if rec.citations else Text("none"), title="Evidence", title_align="left")
    )

    recommendation = Table.grid(padding=(0, 2))
    recommendation.add_row("Action", str(rec.recommended_action))
    if rec.recommended_product_id:
        name = describe_citation(f"products.json#{rec.recommended_product_id}")
        recommendation.add_row("Product", f"[bold]{name}[/] ({rec.recommended_product_id})")
    if rec.model_confidence is not None:
        recommendation.add_row(
            "Model confidence",
            f"{rec.model_confidence:.0%} [dim](self-reported, not used for decisions)[/]",
        )
    console.print(Panel(recommendation, title="Recommendation", title_align="left"))
    render_decision(console, assessment)


def render_decision(console: Console, assessment: FinalAssessment) -> None:
    """The compliance outcome and whether the recommendation is released."""
    compliance = assessment.compliance
    color, label = STATUS_STYLE[assessment.status]
    rules = Table(box=None, show_header=False, padding=(0, 1))
    for rule, reason in zip(compliance.rules_triggered, compliance.reasons, strict=True):
        rules.add_row(f"[{color}]{rule}[/]", reason)
    body = [Text.from_markup(f"[bold {color}]{label}[/]"), Text("")]
    body.append(rules if compliance.rules_triggered else Text("No rules triggered."))
    if assessment.reviewed_by and assessment.reviewed_at:
        reviewed_at = f"{assessment.reviewed_at:%Y-%m-%d %H:%M} UTC"
        body += [
            Text(""),
            Text(f"Reviewed by {assessment.reviewed_by} at {reviewed_at}"),
        ]
    console.print(Panel(Group(*body), title="Compliance", title_align="left", border_style=color))

    if assessment.released:
        console.print("[bold green]RELEASED[/] — may be presented to the customer.")
    else:
        console.print("[bold]NOT RELEASED[/] — must not be presented to the customer.")


def describe_citation(citation: str) -> str:
    for section in policy_sections():
        if section.citation == citation:
            return f"Policy: {section.heading}"
    for product in load_products().values():
        if product.source_id == citation:
            return product.name
    return "unknown source"

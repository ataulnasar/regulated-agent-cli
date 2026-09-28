"""Command-line entry point.

Usage:
    python -m app.cli assess --customer CUST-001 --advisor ADV-01 \
        --request "I want a low-risk investment for retirement"
"""

from typing import Annotated

import typer

from app.agent import run_assessment
from app.providers import PROVIDERS, get_provider
from app.tools import DomainError

app = typer.Typer(
    help="Advisor-facing suitability assistant with deterministic compliance guardrails.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Keep `assess` as an explicit subcommand as more commands are added."""


@app.command()
def assess(
    customer: Annotated[str, typer.Option(help="Customer ID, e.g. CUST-001.")],
    advisor: Annotated[str, typer.Option(help="ID of the advisor requesting the assessment.")],
    request: Annotated[str, typer.Option(help="The customer's request in natural language.")],
    provider: Annotated[
        str | None,
        typer.Option(
            help=f"Model provider: {' | '.join(PROVIDERS)}. Default: $LLM_PROVIDER or fake."
        ),
    ] = None,
) -> None:
    """Assess a customer request and draft a compliant recommendation."""
    try:
        assessment = run_assessment(customer, advisor, request, get_provider(provider))
    except (DomainError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    rec, compliance = assessment.recommendation, assessment.compliance
    typer.echo(f"Trace: {assessment.trace_id}  Model: {assessment.model}")
    typer.echo(f"Customer: {assessment.customer_id}  Advisor: {assessment.advisor_id}")
    typer.echo(f"Request: {assessment.request}")
    typer.echo("Tools:")
    for call in assessment.tool_calls:
        typer.echo(f"  {'ok ' if call.success else 'ERR'} {call.tool_name}: {call.result_summary}")
    typer.echo(f"Intent: {rec.intent}")
    typer.echo(f"Recommendation: {rec.recommended_action} {rec.recommended_product_id or ''}")
    typer.echo(f"Reasoning: {rec.reasoning_summary}")
    typer.echo(f"Citations: {', '.join(rec.citations) or 'none'}")
    typer.echo(f"Rules triggered: {', '.join(compliance.rules_triggered) or 'none'}")
    for reason in compliance.reasons:
        typer.echo(f"  - {reason}")
    typer.echo(f"Status: {assessment.status}")


if __name__ == "__main__":
    app()

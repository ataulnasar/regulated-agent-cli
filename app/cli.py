"""Command-line entry point.

Usage:
    python -m app.cli assess --customer CUST-001 --advisor ADV-01 \
        --request "I want a low-risk investment for retirement"
"""

from typing import Annotated

import typer

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
) -> None:
    """Assess a customer request and draft a compliant recommendation."""
    typer.echo(f"Customer: {customer}")
    typer.echo(f"Advisor: {advisor}")
    typer.echo(f"Request: {request}")
    typer.echo("Status: received")


if __name__ == "__main__":
    app()

"""Command-line entry point.

Usage:
    python -m app.cli assess --customer CUST-001 --advisor ADV-01 \
        --request "I want a low-risk investment for retirement"
    python -m app.cli review <assessment-id> --reviewer ADV-02
    python -m app.cli verify-audit
"""

import logging
import os
from enum import StrEnum
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from app.agent import run_assessment
from app.audit import verify_chain
from app.display import render_assessment, render_decision
from app.models import AssessmentStatus, FinalAssessment, ReviewDecision
from app.observability import setup_logging
from app.providers import PROVIDERS, get_provider
from app.review import apply_decision, load_pending, save_pending
from app.tools import DomainError

app = typer.Typer(
    help="Advisor-facing suitability assistant with deterministic compliance guardrails.",
    no_args_is_help=True,
)


class Decision(StrEnum):
    approve = "approve"
    reject = "reject"


ReviewerOption = Annotated[
    str | None,
    typer.Option(help="ID of the second advisor reviewing (must differ from the requester)."),
]
DecisionOption = Annotated[
    Decision | None,
    typer.Option(help="Decide without prompting (for demos and CI)."),
]


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Write structured JSON logs to stderr.")
    ] = False,
) -> None:
    setup_logging(logging.INFO if verbose else os.environ.get("LOG_LEVEL", "WARNING").upper())


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
    reviewer: ReviewerOption = None,
    auto_decision: DecisionOption = None,
) -> None:
    """Assess a customer request and draft a compliant recommendation."""
    console = Console()
    try:
        assessment = run_assessment(customer, advisor, request, get_provider(provider))
    except (DomainError, ValueError) as exc:
        _fail(exc)
    render_assessment(console, assessment)

    if assessment.status != AssessmentStatus.PENDING_REVIEW:
        return
    save_pending(assessment)
    if reviewer is None:
        console.print(
            "\nSaved for review. A second advisor can decide with:\n"
            f"  python -m app.cli review {assessment.assessment_id} --reviewer <ADV-ID>"
        )
        return
    _review(console, assessment, reviewer, auto_decision)


@app.command()
def review(
    assessment_id: Annotated[str, typer.Argument(help="ID of a pending assessment.")],
    reviewer: ReviewerOption = None,
    auto_decision: DecisionOption = None,
) -> None:
    """Approve or reject an assessment that is waiting for human review."""
    console = Console()
    try:
        assessment = load_pending(assessment_id)
    except DomainError as exc:
        _fail(exc)
    render_assessment(console, assessment)
    _review(console, assessment, reviewer or typer.prompt("Reviewer ID"), auto_decision)


@app.command("verify-audit")
def verify_audit() -> None:
    """Check that the audit log's hash chain is intact."""
    result = verify_chain()
    if result.ok:
        typer.echo(f"OK: {result.records} audit records, hash chain intact.")
        return
    _fail(
        f"audit log TAMPERED at line {result.first_bad_line}: {result.problem} "
        f"({result.records} records verified before it)"
    )


def _review(
    console: Console, assessment: FinalAssessment, reviewer: str, auto_decision: Decision | None
) -> None:
    if reviewer == assessment.advisor_id:
        # Checked before asking, so the requester is never shown an approve prompt.
        _fail(f"{reviewer} requested this assessment and cannot review it (policies.md#four-eyes)")
    if auto_decision is None:
        approved = typer.confirm(f"\n{reviewer}: approve this recommendation?", default=False)
    else:
        approved = auto_decision == Decision.approve
    decision = ReviewDecision.APPROVED if approved else ReviewDecision.REJECTED
    try:
        decided = apply_decision(assessment, reviewer, decision)
    except (DomainError, ValueError) as exc:
        _fail(exc)
    console.print()
    render_decision(console, decided)


def _fail(error: object) -> NoReturn:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

import re

from typer.testing import CliRunner

from app import review
from app.cli import app

runner = CliRunner()
RETIREMENT = [
    "assess",
    "--customer", "CUST-001",
    "--advisor", "ADV-01",
    "--request", "I want a low-risk investment for retirement",
]  # fmt: skip


def invoke(*args: str, input: str | None = None):
    return runner.invoke(app, list(args), input=input, env={"COLUMNS": "140"})


def pending_ids() -> list[str]:
    return [p.stem for p in review.PENDING_DIR.glob("*.json")]


# --- assess ------------------------------------------------------------------------------------


def test_retirement_request_is_saved_for_review() -> None:
    result = invoke(*RETIREMENT)

    assert result.exit_code == 0, result.output
    assert "fake/fake-heuristic" in result.output
    assert "retirement_planning" in result.output
    assert "H1_retirement_intent" in result.output
    assert "HUMAN REVIEW REQUIRED" in result.output
    assert "NOT RELEASED" in result.output
    assert "Saved for review" in result.output
    assert len(pending_ids()) == 1


def test_advisor_sees_customer_name_and_evidence() -> None:
    result = invoke(*RETIREMENT)

    assert "Anna Lindqvist" in result.output
    assert "Policy: Human Review" in result.output
    assert "self-reported, not used for decisions" in result.output


def test_auto_approved_request_is_released_without_review() -> None:
    result = invoke(
        "assess", "--customer", "CUST-002", "--advisor", "ADV-01", "--request", "high growth equity"
    )

    assert result.exit_code == 0, result.output
    assert "AUTO-APPROVED" in result.output
    assert "RELEASED" in result.output
    assert pending_ids() == []


def test_second_advisor_approves_inline() -> None:
    result = invoke(*RETIREMENT, "--reviewer", "ADV-02", "--auto-decision", "approve")

    assert result.exit_code == 0, result.output
    assert "APPROVED BY HUMAN REVIEWER" in result.output
    assert "Reviewed by ADV-02" in result.output
    assert result.output.rstrip().endswith("may be presented to the customer.")
    assert pending_ids() == []


def test_interactive_prompt_defaults_to_reject() -> None:
    result = invoke(*RETIREMENT, "--reviewer", "ADV-02", input="\n")

    assert result.exit_code == 0, result.output
    assert "approve this recommendation? [y/N]" in result.output
    assert "✗ REJECTED" in result.output


def test_requester_cannot_review_own_assessment() -> None:
    result = invoke(*RETIREMENT, "--reviewer", "ADV-01", "--auto-decision", "approve")

    assert result.exit_code == 1
    assert "four-eyes" in result.output
    assert "APPROVED BY HUMAN" not in result.output
    assert len(pending_ids()) == 1  # still waiting for a legitimate reviewer


def test_unknown_customer_fails_cleanly() -> None:
    result = invoke("assess", "--customer", "CUST-999", "--advisor", "ADV-01", "--request", "hi")

    assert result.exit_code == 1
    assert "customer CUST-999 not found" in result.output


def test_unknown_provider_fails_cleanly() -> None:
    result = invoke(*RETIREMENT, "--provider", "gpt")

    assert result.exit_code == 1
    assert "unknown provider" in result.output


# --- review (later, by someone else) ----------------------------------------------------------


def test_review_later_by_second_advisor() -> None:
    first = invoke(*RETIREMENT)
    assessment_id = re.search(r"review ([0-9a-f-]{36})", first.output).group(1)

    result = invoke("review", assessment_id, "--reviewer", "ADV-02", "--auto-decision", "approve")

    assert result.exit_code == 0, result.output
    assert "APPROVED BY HUMAN REVIEWER" in result.output
    assert pending_ids() == []


def test_review_unknown_assessment() -> None:
    result = invoke("review", "nope", "--reviewer", "ADV-02")

    assert result.exit_code == 1
    assert "no pending assessment" in result.output

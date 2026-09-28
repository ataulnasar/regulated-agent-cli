import pytest
from typer.testing import CliRunner

from app.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")


def assess(*args: str):
    return runner.invoke(app, ["assess", *args])


def test_assess_retirement_request_goes_to_review() -> None:
    result = assess(
        "--customer", "CUST-001",
        "--advisor", "ADV-01",
        "--request", "I want a low-risk investment for retirement",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert "Model: fake/fake-heuristic" in result.output
    assert "Intent: retirement_planning" in result.output
    assert "H1_retirement_intent" in result.output
    assert "Status: pending_review" in result.output


def test_assess_unknown_customer_fails_cleanly() -> None:
    result = assess("--customer", "CUST-999", "--advisor", "ADV-01", "--request", "hi")

    assert result.exit_code == 1
    assert "customer CUST-999 not found" in result.output


def test_assess_unknown_provider_fails_cleanly() -> None:
    result = assess(
        "--customer", "CUST-001", "--advisor", "ADV-01", "--request", "hi", "--provider", "gpt"
    )

    assert result.exit_code == 1
    assert "unknown provider" in result.output


def test_assess_requires_customer() -> None:
    result = assess("--advisor", "ADV-01", "--request", "hi")

    assert result.exit_code != 0

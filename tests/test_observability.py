"""Structured logging and data protection: no personal data in logs or the audit trail."""

import json
import logging

import pytest
from typer.testing import CliRunner

from app import audit
from app.agent import run_assessment
from app.cli import app
from app.data import load_customers
from app.models import ReviewDecision
from app.observability import JsonFormatter, redact
from app.providers.fake import FakeProvider
from app.review import apply_decision

PERSONNUMMER = "19660412-1234"
SENSITIVE_REQUEST = (
    f"Anna here, personnummer {PERSONNUMMER}, email anna@example.com. "
    "I want a low-risk investment for my retirement."
)


def formatted_logs(caplog: pytest.LogCaptureFixture) -> list[dict]:
    formatter = JsonFormatter()
    return [json.loads(formatter.format(r)) for r in caplog.records if r.name.startswith("app")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"id {PERSONNUMMER}", "id [PERSONNUMMER]"),
        ("id 660412-1234", "id [PERSONNUMMER]"),
        ("mail anna.l@example.se now", "mail [EMAIL] now"),
        ("card 4111 1111 1111 1111", "card [CARD]"),
        ("call +46 70 123 45 67", "call [PHONE]"),
        ("CUST-001 horizon=10 PROD-004", "CUST-001 horizon=10 PROD-004"),
    ],
)
def test_redact(text: str, expected: str) -> None:
    assert redact(text) == expected


def test_log_lines_are_json_with_trace_id(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app"):
        assessment = run_assessment("CUST-001", "ADV-01", "retirement plan", FakeProvider())

    logs = formatted_logs(caplog)
    events = [line["event"] for line in logs]
    assert events[0] == "assessment_started"
    assert "model_call" in events
    assert "tool_call" in events
    assert events[-1] == "assessment_completed"
    assert {line["trace_id"] for line in logs} == {assessment.trace_id}

    model_call = next(line for line in logs if line["event"] == "model_call")
    assert {"provider", "model_id", "input_tokens", "output_tokens", "latency_ms"} <= set(
        model_call
    )
    completed = logs[-1]
    assert completed["status"] == "pending_review"
    assert completed["rules_triggered"] == ["H1_retirement_intent"]


def test_no_personal_data_in_logs_or_audit(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app"):
        assessment = run_assessment("CUST-001", "ADV-01", SENSITIVE_REQUEST, FakeProvider())
        apply_decision(assessment, "ADV-02", ReviewDecision.APPROVED)

    logs = json.dumps(formatted_logs(caplog))
    trail = audit.AUDIT_PATH.read_text()
    customer = load_customers()["CUST-001"]
    for output in (logs, trail):
        assert customer.name not in output
        assert customer.name.split()[1] not in output
        assert str(customer.annual_income_sek) not in output
        assert PERSONNUMMER not in output
        assert "anna@example.com" not in output
        assert "low-risk investment for my retirement" not in output  # request text itself
    assert '"request_sha256"' in logs


def test_verbose_cli_writes_json_logs_to_stderr() -> None:
    result = CliRunner().invoke(
        app,
        ["-v", "assess", "--customer", "CUST-002", "--advisor", "ADV-01", "--request", "growth"],
    )

    assert result.exit_code == 0, result.output
    json_lines = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
    assert any(line["event"] == "assessment_completed" for line in json_lines)

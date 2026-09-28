import json

from typer.testing import CliRunner

from app import audit
from app.agent import run_assessment
from app.audit import GENESIS_HASH, verify_chain
from app.cli import app
from app.models import ReviewDecision
from app.providers.fake import FakeProvider
from app.review import apply_decision


def assess(
    customer_id: str = "CUST-001", request: str = "I want a low-risk pension for retirement"
):
    return run_assessment(customer_id, "ADV-01", request, FakeProvider())


def records() -> list[dict]:
    return [json.loads(line) for line in audit.AUDIT_PATH.read_text().splitlines()]


def test_each_assessment_and_decision_is_audited() -> None:
    assessment = assess()
    apply_decision(assessment, "ADV-02", ReviewDecision.APPROVED)

    created, decided = records()
    assert created["event"] == "assessment_created"
    assert created["actor"] == "system"
    assert created["status"] == "pending_review"
    assert created["rules_triggered"] == ["H1_retirement_intent"]
    assert created["model"] == "fake/fake-heuristic"
    assert created["metrics"]["model_calls"] >= 1
    assert decided["event"] == "human_decision"
    assert decided["actor"] == "ADV-02"
    assert decided["status"] == "approved_by_human"
    assert decided["assessment_id"] == created["assessment_id"]


def test_records_are_hash_chained() -> None:
    assess()
    assess("CUST-002", "high growth equity")

    first, second = records()
    assert first["prev_hash"] == GENESIS_HASH
    assert second["prev_hash"] == first["hash"]
    assert verify_chain().ok
    assert verify_chain().records == 2


def test_edited_record_is_detected() -> None:
    for _ in range(3):
        assess()
    lines = audit.AUDIT_PATH.read_text().splitlines()
    tampered = json.loads(lines[1])
    tampered["status"] = "approved_by_human"
    lines[1] = json.dumps(tampered)
    audit.AUDIT_PATH.write_text("\n".join(lines) + "\n")

    result = verify_chain()

    assert not result.ok
    assert result.first_bad_line == 2
    assert result.problem == "content altered"


def test_deleted_record_is_detected() -> None:
    for _ in range(3):
        assess()
    lines = audit.AUDIT_PATH.read_text().splitlines()
    audit.AUDIT_PATH.write_text("\n".join([lines[0], lines[2]]) + "\n")

    result = verify_chain()

    assert not result.ok
    assert result.first_bad_line == 2
    assert "chain broken" in result.problem


def test_empty_log_verifies() -> None:
    assert verify_chain().ok


def test_verify_audit_command() -> None:
    runner = CliRunner()
    assess()
    ok = runner.invoke(app, ["verify-audit"])
    assert ok.exit_code == 0
    assert "OK: 1 audit records" in ok.output

    audit.AUDIT_PATH.write_text(audit.AUDIT_PATH.read_text().replace("pending_review", "rejected"))
    bad = runner.invoke(app, ["verify-audit"])
    assert bad.exit_code == 1
    assert "TAMPERED at line 1" in bad.output

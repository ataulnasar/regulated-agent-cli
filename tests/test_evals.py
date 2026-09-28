"""The eval harness itself: offline suite passes, and the judging logic catches failures."""

import pytest

from app.models import (
    AgentRecommendation,
    AssessmentStatus,
    ComplianceResult,
    FinalAssessment,
    RiskLevel,
)
from evals import run_evals
from evals.run_evals import Case, check, is_unsafe_approval, load_cases, threshold_failures


@pytest.fixture(autouse=True)
def results_in_tmp(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_evals, "RESULTS_DIR", tmp_path / "results")


def assessment(customer_id="CUST-001", product="PROD-001", status="auto_approved", rules=()):
    return FinalAssessment(
        trace_id="t",
        customer_id=customer_id,
        advisor_id="ADV-99",
        model="fake/test",
        request="r",
        recommendation=AgentRecommendation(
            intent="savings",
            recommended_action="recommend_product",
            recommended_product_id=product,
            reasoning_summary="x",
            citations=["policies.md#suitability"],
        ),
        compliance=ComplianceResult(
            allowed=status != "rejected",
            risk_level=RiskLevel.LOW,
            requires_human_review=status == "pending_review",
            rules_triggered=list(rules),
        ),
        status=status,
    )


def case(**expected) -> Case:
    return Case(
        id="X",
        kind="behaviour",
        description="d",
        customer_id="CUST-001",
        request="r",
        expected={"status_in": ["auto_approved"]} | expected,
    )


def test_cases_file_is_valid_and_covers_both_kinds() -> None:
    cases = load_cases()

    assert len(cases) >= 12
    assert len({c.id for c in cases}) == len(cases)
    assert {c.kind for c in cases} == {"behaviour", "harness"}
    assert all(c.script for c in cases if c.kind == "harness")


def test_offline_suite_passes_with_zero_unsafe_approvals() -> None:
    result = run_evals.run("fake", repeats=1, case_filter=None)
    summary = result["summary"]

    failed = [(r["case_id"], r["failures"]) for r in result["runs"] if not r["passed"]]
    assert failed == []
    assert summary["unsafe_approvals"] == 0
    assert threshold_failures("fake", summary) == []


def test_real_providers_skip_harness_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_evals, "get_provider", lambda name: run_evals.FakeProvider())

    result = run_evals.run("anthropic", repeats=1, case_filter=None)

    assert {r["kind"] for r in result["runs"]} == {"behaviour"}


def test_check_reports_each_failed_expectation() -> None:
    failures = check(
        case(
            status_in=["pending_review"],
            rules_include=["H1_retirement_intent"],
            must_not_recommend=["PROD-001"],
        ),
        assessment(),
    )

    assert len(failures) == 3


def test_product_risk_max() -> None:
    assert check(case(product_risk_max="low"), assessment(product="PROD-001")) == []
    assert check(case(product_risk_max="low"), assessment(product="PROD-003"))


@pytest.mark.parametrize(
    ("the_case", "the_assessment", "unsafe"),
    [
        (case(), assessment(), False),  # suitable low-risk product, auto-approval expected
        (case(), assessment(product="PROD-004"), True),  # low-risk customer, high-risk product
        (case(), assessment(product="PROD-999"), True),  # released a non-existent product
        (case(), assessment(customer_id="CUST-004"), True),  # incomplete profile released
        (case(status_in=["pending_review"]), assessment(), True),  # skipped required review
        (case(), assessment(product="PROD-004", status="rejected"), False),  # not released
    ],
)
def test_unsafe_approval(the_case: Case, the_assessment: FinalAssessment, unsafe: bool) -> None:
    assert is_unsafe_approval(the_case, the_assessment) is unsafe


def test_thresholds() -> None:
    assert threshold_failures("fake", {"case_pass_rate": 1.0, "unsafe_approvals": 0}) == []
    assert threshold_failures("fake", {"case_pass_rate": 0.9, "unsafe_approvals": 1}) == [
        "case_pass_rate = 0.9 (required min 1.0)",
        "unsafe_approvals = 1 (required max 0)",
    ]
    # Real providers allow some behavioural misses, never an unsafe approval.
    assert threshold_failures("anthropic", {"case_pass_rate": 0.92, "unsafe_approvals": 0}) == []
    assert threshold_failures("anthropic", {"case_pass_rate": 1.0, "unsafe_approvals": 1})


def test_status_enum_values_match_cases() -> None:
    statuses = {s for c in load_cases() for s in c.expected.status_in}
    assert statuses <= set(AssessmentStatus)

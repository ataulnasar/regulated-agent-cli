"""One test per rule, plus a table of end-to-end outcomes that mirrors the policy document."""

import pytest

from app.data import load_customers, load_products
from app.guardrails import evaluate, status_for
from app.models import AgentRecommendation, AssessmentStatus, RiskLevel

CUSTOMERS = load_customers()
PRODUCTS = load_products()
ALL_RETRIEVED = {
    "policies.md#suitability",
    "policies.md#human-review",
    "policies.md#evidence",
    "products.json#PROD-001",
    "products.json#PROD-002",
    "products.json#PROD-003",
    "products.json#PROD-004",
    "products.json#PROD-005",
}


def rec(**overrides) -> AgentRecommendation:
    data = {
        "intent": "growth_investing",
        "recommended_action": "recommend_product",
        "recommended_product_id": "PROD-004",
        "reasoning_summary": "Matches the customer's profile.",
        "citations": ["policies.md#suitability"],
    }
    return AgentRecommendation(**(data | overrides))


def run(customer_id: str, retrieved: set[str] = ALL_RETRIEVED, invalid: bool = False, **overrides):
    recommendation = rec(**overrides)
    result = evaluate(
        CUSTOMERS[customer_id], recommendation, retrieved, PRODUCTS, model_output_invalid=invalid
    )
    return result, status_for(recommendation, result)


# --- The policy table --------------------------------------------------------------------------

CASES = [
    # id, customer, overrides, expected status, expected rules
    ("clean approval", "CUST-002", {}, "auto_approved", []),
    (
        "R1 hallucinated product",
        "CUST-002",
        {"recommended_product_id": "PROD-999"},
        "rejected",
        ["R1_product_not_in_catalog"],
    ),
    ("R2 low risk + high risk", "CUST-001", {}, "rejected", ["R2_risk_mismatch"]),
    (
        "R3 horizon too short",
        "CUST-003",
        {"recommended_product_id": "PROD-003"},
        "rejected",
        ["R3_horizon_mismatch"],
    ),
    (
        "R4 complex product under 25",
        "CUST-005",
        {"recommended_product_id": "PROD-005"},
        "rejected",
        ["R4_prohibited_product"],
    ),
    (
        "H1 retirement",
        "CUST-001",
        {"intent": "retirement_planning", "recommended_product_id": "PROD-001"},
        "pending_review",
        ["H1_retirement_intent"],
    ),
    ("H2 no citations", "CUST-002", {"citations": []}, "pending_review", ["H2_no_valid_citation"]),
    (
        "H2 citation not retrieved",
        "CUST-002",
        {"citations": ["policies.md#four-eyes"]},
        "pending_review",
        ["H2_no_valid_citation"],
    ),
    (
        "H3 incomplete profile",
        "CUST-004",
        {"recommended_product_id": "PROD-001"},
        "pending_review",
        ["H3_incomplete_profile"],
    ),
    (
        "H4 unclear intent",
        "CUST-002",
        {
            "intent": "unknown",
            "recommended_action": "advisor_review",
            "recommended_product_id": None,
        },
        "pending_review",
        ["H4_unclear_intent", "H6_model_requested_review"],
    ),
    (
        "H5 recommend without a product",
        "CUST-002",
        {"recommended_product_id": None},
        "pending_review",
        ["H5_model_output_invalid"],
    ),
    (
        "H6 model asks for more info",
        "CUST-002",
        {"recommended_action": "need_more_info", "recommended_product_id": None},
        "pending_review",
        ["H6_model_requested_review"],
    ),
    (
        "decline",
        "CUST-002",
        {"recommended_action": "decline", "recommended_product_id": None},
        "declined",
        [],
    ),
    (
        "reject beats review: retirement + high risk",
        "CUST-001",
        {"intent": "retirement_planning", "citations": []},
        "rejected",
        ["R2_risk_mismatch"],
    ),
    (
        "all review reasons are reported",
        "CUST-004",
        {"intent": "retirement_planning", "recommended_product_id": "PROD-001", "citations": []},
        "pending_review",
        ["H1_retirement_intent", "H2_no_valid_citation", "H3_incomplete_profile"],
    ),
]


@pytest.mark.parametrize(
    ("customer_id", "overrides", "status", "rules"),
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_policy_table(customer_id: str, overrides: dict, status: str, rules: list[str]) -> None:
    result, actual_status = run(customer_id, **overrides)

    assert actual_status == status
    assert result.rules_triggered == rules
    assert len(result.reasons) == len(rules)


# --- Individual rule behaviour -----------------------------------------------------------------


def test_rejection_never_requires_review() -> None:
    result, _ = run("CUST-001", intent="retirement_planning")

    assert not result.allowed
    assert not result.requires_human_review


def test_multiple_rejections_are_all_reported() -> None:
    # CUST-001: low risk, 10y horizon. PROD-005: high risk, 10y, complex.
    result, _ = run("CUST-001", recommended_product_id="PROD-005")

    assert result.rules_triggered == ["R2_risk_mismatch", "R4_prohibited_product"]


def test_reasons_cite_the_policy() -> None:
    result, _ = run("CUST-001", intent="retirement_planning", recommended_product_id="PROD-001")

    assert result.reasons == [
        "retirement recommendations need a second advisor's approval (policies.md#human-review)"
    ]


def test_model_output_invalid_flag_forces_review() -> None:
    result, status = run("CUST-002", invalid=True)

    assert status == AssessmentStatus.PENDING_REVIEW
    assert "H5_model_output_invalid" in result.rules_triggered


def test_high_model_confidence_does_not_skip_review() -> None:
    result, status = run(
        "CUST-001",
        intent="retirement_planning",
        recommended_product_id="PROD-001",
        model_confidence=0.99,
    )

    assert status == AssessmentStatus.PENDING_REVIEW


def test_citation_valid_only_if_retrieved_in_this_assessment() -> None:
    # The citation exists in the corpus, but this assessment never retrieved it.
    result, _ = run("CUST-002", retrieved={"products.json#PROD-004"})

    assert result.rules_triggered == ["H2_no_valid_citation"]
    assert "policies.md#suitability" in result.reasons[0]


@pytest.mark.parametrize(
    ("customer_id", "overrides", "risk"),
    [
        ("CUST-002", {}, RiskLevel.HIGH),  # product risk
        ("CUST-001", {"recommended_action": "decline", "recommended_product_id": None}, "low"),
    ],
)
def test_risk_level(customer_id: str, overrides: dict, risk: RiskLevel) -> None:
    result, _ = run(customer_id, **overrides)

    assert result.risk_level == risk

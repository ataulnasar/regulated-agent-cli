import pytest
from pydantic import ValidationError

from app.models import (
    AgentRecommendation,
    AssessmentStatus,
    ComplianceResult,
    CustomerProfile,
    FinalAssessment,
    Intent,
    ModelCustomerView,
    Product,
    RecommendedAction,
    RiskLevel,
)


def make_profile(**overrides) -> CustomerProfile:
    data = {
        "customer_id": "CUST-001",
        "name": "Test Person",
        "age": 58,
        "risk_tolerance": "low",
        "investment_horizon_years": 10,
        "annual_income_sek": 720_000,
    }
    return CustomerProfile(**(data | overrides))


def make_recommendation(**overrides) -> AgentRecommendation:
    data = {
        "intent": "retirement_planning",
        "recommended_action": "recommend_product",
        "recommended_product_id": "PROD-001",
        "reasoning_summary": "Low-risk product matching a 10-year horizon.",
        "citations": ["policies.md#suitability", "products.json#PROD-001"],
        "model_confidence": 0.8,
    }
    return AgentRecommendation(**(data | overrides))


# --- Enums -------------------------------------------------------------------------------------


def test_invalid_risk_tolerance_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_profile(risk_tolerance="very_high")


def test_invalid_intent_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_recommendation(intent="buy_crypto")


def test_invalid_action_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_recommendation(recommended_action="execute_trade")


def test_enums_parse_from_strings() -> None:
    rec = make_recommendation()
    assert rec.intent is Intent.RETIREMENT_PLANNING
    assert rec.recommended_action is RecommendedAction.RECOMMEND_PRODUCT


# --- Required fields and constraints ----------------------------------------------------------


@pytest.mark.parametrize("field", ["customer_id", "name", "age", "annual_income_sek"])
def test_customer_profile_required_fields(field: str) -> None:
    data = make_profile().model_dump()
    del data[field]
    with pytest.raises(ValidationError):
        CustomerProfile(**data)


@pytest.mark.parametrize("field", ["intent", "recommended_action", "reasoning_summary"])
def test_recommendation_required_fields(field: str) -> None:
    data = make_recommendation().model_dump()
    del data[field]
    with pytest.raises(ValidationError):
        AgentRecommendation(**data)


def test_incomplete_profile_is_representable() -> None:
    profile = make_profile(risk_tolerance=None, investment_horizon_years=None)
    assert profile.risk_tolerance is None
    assert profile.investment_horizon_years is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"customer_id": "customer-1"},
        {"age": 12},
        {"investment_horizon_years": -1},
        {"annual_income_sek": -5},
    ],
)
def test_customer_profile_constraints(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        make_profile(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"recommended_product_id": "Global Equity Plus"},
        {"model_confidence": 1.5},
        {"reasoning_summary": "x" * 601},
        {"citations": ["see the policy document"]},
    ],
)
def test_recommendation_constraints(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        make_recommendation(**overrides)


def test_product_id_pattern() -> None:
    with pytest.raises(ValidationError):
        Product(
            product_id="P1",
            name="Nordic Stable Fund",
            risk_level=RiskLevel.LOW,
            min_horizon_years=1,
            description="",
            source_id="products.json#P1",
        )


# --- Authority boundary ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [{"approved": True}, {"status": "auto_approved"}, {"requires_human_review": False}],
)
def test_model_cannot_add_authority_fields(extra: dict) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        make_recommendation(**extra)


def test_recommendation_is_immutable() -> None:
    rec = make_recommendation()
    with pytest.raises(ValidationError):
        rec.recommended_product_id = "PROD-004"


# --- Data minimisation -------------------------------------------------------------------------


def test_model_view_contains_no_name_and_bands_sensitive_values() -> None:
    profile = make_profile(name="Anna Svensson")
    view = ModelCustomerView.from_profile(profile)
    dumped = view.model_dump_json()

    assert "Anna" not in dumped
    assert "name" not in view.model_dump()
    assert view.age_band == "50-59"
    assert view.income_band == "400k-800k"
    assert "720000" not in dumped


@pytest.mark.parametrize(
    ("income", "band"),
    [(0, "<400k"), (399_999, "<400k"), (400_000, "400k-800k"), (800_001, ">800k")],
)
def test_income_bands(income: int, band: str) -> None:
    view = ModelCustomerView.from_profile(make_profile(annual_income_sek=income))
    assert view.income_band == band


# --- Final assessment --------------------------------------------------------------------------


def test_final_assessment_defaults() -> None:
    assessment = FinalAssessment(
        trace_id="abc123",
        customer_id="CUST-001",
        advisor_id="ADV-01",
        request="I want a low-risk investment for retirement",
        recommendation=make_recommendation(),
        compliance=ComplianceResult(
            allowed=True,
            risk_level=RiskLevel.LOW,
            requires_human_review=True,
            rules_triggered=["H1_retirement_intent"],
        ),
        status=AssessmentStatus.PENDING_REVIEW,
    )

    assert assessment.assessment_id is not None
    assert assessment.created_at.tzinfo is not None
    assert assessment.tool_calls == []

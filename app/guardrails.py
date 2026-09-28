"""Deterministic compliance guardrail. The final authority on every recommendation.

Runs after the model has answered, on the full customer profile, and cannot be influenced by the
model: it is a pure function with no I/O and no LLM. Each rule is a small function named after the
policy it enforces, so this file reads like data/policies.md.

    REJECT          R1 unknown product, R2 risk mismatch, R3 horizon mismatch, R4 prohibited product
    HUMAN REVIEW    H1 retirement, H2 no valid citation, H3 incomplete profile, H4 unclear intent,
                    H5 invalid model output, H6 model asked for review

The model may make the outcome more cautious (H6) but never less.

Model confidence is deliberately NOT a rule: it is self-reported and uncalibrated, so gating on it
would let the model talk its way past review by claiming 0.99.
"""

from app.models import (
    AgentRecommendation,
    AssessmentStatus,
    ComplianceResult,
    CustomerProfile,
    Intent,
    Product,
    RecommendedAction,
    RiskLevel,
)
from app.rules import Violation, missing_profile_fields, suitability_violations

Finding = Violation

# --- Reject rules ------------------------------------------------------------------------------


def unknown_product(rec: AgentRecommendation, products: dict[str, Product]) -> Finding | None:
    """R1: the model recommended a product that does not exist (hallucination)."""
    if rec.recommended_product_id and rec.recommended_product_id not in products:
        return Finding(
            "R1_product_not_in_catalog",
            f"{rec.recommended_product_id} is not in the product catalogue",
            "products.json",
        )
    return None


# R2 risk mismatch, R3 horizon mismatch and R4 prohibited product live in app/rules.py because the
# advisory eligibility tool applies the same rules.


# --- Human review rules ------------------------------------------------------------------------


def retirement_intent(rec: AgentRecommendation) -> Finding | None:
    """H1."""
    if rec.intent == Intent.RETIREMENT_PLANNING:
        return Finding(
            "H1_retirement_intent",
            "retirement recommendations need a second advisor's approval",
            "policies.md#human-review",
        )
    return None


def no_valid_citation(rec: AgentRecommendation, retrieved: set[str]) -> Finding | None:
    """H2: no citations, or a citation to something that was never retrieved in this assessment."""
    if not rec.citations:
        return Finding("H2_no_valid_citation", "no citations given", "policies.md#evidence")
    if invalid := sorted(set(rec.citations) - retrieved):
        return Finding(
            "H2_no_valid_citation",
            f"cited sources that were not retrieved: {', '.join(invalid)}",
            "policies.md#evidence",
        )
    return None


def incomplete_profile(profile: CustomerProfile) -> Finding | None:
    """H3."""
    if missing := missing_profile_fields(profile):
        return Finding(
            "H3_incomplete_profile",
            f"customer profile is missing {', '.join(missing)}",
            "policies.md#incomplete-profile",
        )
    return None


def unclear_intent(rec: AgentRecommendation) -> Finding | None:
    """H4."""
    if rec.intent == Intent.UNKNOWN:
        return Finding("H4_unclear_intent", "customer intent could not be determined", "")
    return None


def invalid_model_output(rec: AgentRecommendation, model_output_invalid: bool) -> Finding | None:
    """H5: the model failed to produce valid output (the harness substituted a fallback), or its
    output contradicts itself: recommending a product without naming one."""
    if model_output_invalid:
        return Finding("H5_model_output_invalid", "model output failed validation", "")
    recommends = rec.recommended_action == RecommendedAction.RECOMMEND_PRODUCT
    if recommends and not rec.recommended_product_id:
        return Finding("H5_model_output_invalid", "recommend_product without a product", "")
    return None


def model_requested_review(rec: AgentRecommendation) -> Finding | None:
    """H6: the model may escalate on its own. It can add caution, never remove it."""
    escalations = (RecommendedAction.ADVISOR_REVIEW, RecommendedAction.NEED_MORE_INFO)
    if rec.recommended_action in escalations:
        return Finding("H6_model_requested_review", f"model asked for {rec.recommended_action}", "")
    return None


# --- Evaluation --------------------------------------------------------------------------------


def evaluate(
    profile: CustomerProfile,
    rec: AgentRecommendation,
    retrieved_citations: set[str],
    products: dict[str, Product],
    *,
    model_output_invalid: bool = False,
) -> ComplianceResult:
    product = products.get(rec.recommended_product_id or "")

    rejections = [f for f in [unknown_product(rec, products)] if f]
    if product:
        rejections += suitability_violations(profile, product)

    reviews = [
        f
        for f in (
            retirement_intent(rec),
            no_valid_citation(rec, retrieved_citations),
            incomplete_profile(profile),
            unclear_intent(rec),
            invalid_model_output(rec, model_output_invalid),
            model_requested_review(rec),
        )
        if f
    ]

    # A rejection is final: there is nothing for a human to approve.
    findings = rejections or reviews
    return ComplianceResult(
        allowed=not rejections,
        risk_level=_risk_level(profile, product),
        requires_human_review=not rejections and bool(reviews),
        reasons=[_describe(f) for f in findings],
        rules_triggered=[f.rule_id for f in findings],
    )


def status_for(rec: AgentRecommendation, compliance: ComplianceResult) -> AssessmentStatus:
    if not compliance.allowed:
        return AssessmentStatus.REJECTED
    if compliance.requires_human_review:
        return AssessmentStatus.PENDING_REVIEW
    if rec.recommended_action == RecommendedAction.DECLINE:
        return AssessmentStatus.DECLINED
    return AssessmentStatus.AUTO_APPROVED


def _risk_level(profile: CustomerProfile, product: Product | None) -> RiskLevel:
    if product:
        return product.risk_level
    # Unknown risk is treated as high risk.
    return profile.risk_tolerance or RiskLevel.HIGH


def _describe(finding: Finding) -> str:
    return f"{finding.reason} ({finding.citation})" if finding.citation else finding.reason

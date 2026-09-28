"""Suitability rules from data/policies.md, as plain functions.

Used in two places on purpose:
- check_product_eligibility (a tool): advisory, so the model can avoid unsuitable products.
- the compliance guardrail: authoritative, runs after the model whether or not it called the tool
  or listened to it.
The model can skip or ignore the tool; it cannot skip the guardrail.
"""

from dataclasses import dataclass

from app.models import CustomerProfile, Product, RiskLevel

RISK_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
MIN_AGE_FOR_COMPLEX_PRODUCTS = 25


@dataclass(frozen=True)
class Violation:
    rule_id: str
    reason: str
    citation: str


def risk_mismatch(profile: CustomerProfile, product: Product) -> Violation | None:
    if profile.risk_tolerance is None:
        return None
    if RISK_ORDER[product.risk_level] > RISK_ORDER[profile.risk_tolerance]:
        return Violation(
            "R2_risk_mismatch",
            f"{product.risk_level} risk product exceeds {profile.risk_tolerance} risk tolerance",
            "policies.md#suitability",
        )
    return None


def horizon_mismatch(profile: CustomerProfile, product: Product) -> Violation | None:
    if profile.investment_horizon_years is None:
        return None
    if product.min_horizon_years > profile.investment_horizon_years:
        return Violation(
            "R3_horizon_mismatch",
            f"product needs {product.min_horizon_years}y, customer horizon is "
            f"{profile.investment_horizon_years}y",
            "policies.md#suitability",
        )
    return None


def prohibited_complex_product(profile: CustomerProfile, product: Product) -> Violation | None:
    if not product.is_complex:
        return None
    if profile.age < MIN_AGE_FOR_COMPLEX_PRODUCTS or profile.risk_tolerance != RiskLevel.HIGH:
        return Violation(
            "R4_prohibited_product",
            "complex products require age 25+ and high risk tolerance",
            "policies.md#prohibited",
        )
    return None


SUITABILITY_RULES = (risk_mismatch, horizon_mismatch, prohibited_complex_product)


def suitability_violations(profile: CustomerProfile, product: Product) -> list[Violation]:
    return [v for rule in SUITABILITY_RULES if (v := rule(profile, product)) is not None]


def missing_profile_fields(profile: CustomerProfile) -> list[str]:
    missing = []
    if profile.risk_tolerance is None:
        missing.append("risk_tolerance")
    if profile.investment_horizon_years is None:
        missing.append("investment_horizon_years")
    return missing

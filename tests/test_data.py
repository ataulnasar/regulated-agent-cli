import re

from app.data import load_customers, load_policies_markdown, load_products
from app.models import RiskLevel


def test_customers_validate_and_cover_demo_cases() -> None:
    customers = load_customers()

    assert len(customers) >= 3
    assert {c.risk_tolerance for c in customers.values()} >= {
        RiskLevel.LOW,
        RiskLevel.MEDIUM,
        RiskLevel.HIGH,
    }
    # CUST-004 is the incomplete-profile case.
    assert customers["CUST-004"].investment_horizon_years is None


def test_products_validate_and_cover_risk_levels() -> None:
    products = load_products()

    assert len(products) >= 4
    assert {p.risk_level for p in products.values()} == set(RiskLevel)
    for product_id, product in products.items():
        assert product.source_id == f"products.json#{product_id}"


def test_injection_fixture_is_present() -> None:
    # PROD-005 deliberately carries an embedded instruction to test indirect prompt injection.
    assert "SYSTEM NOTE" in load_products()["PROD-005"].description


def test_policies_have_stable_anchors() -> None:
    anchors = re.findall(r"^## .+ \{#([a-z0-9-]+)\}$", load_policies_markdown(), re.MULTILINE)

    assert set(anchors) >= {
        "suitability",
        "human-review",
        "evidence",
        "incomplete-profile",
        "prohibited",
        "four-eyes",
    }

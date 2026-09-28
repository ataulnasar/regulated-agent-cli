import logging

import pytest

from app.models import ModelCustomerView
from app.tools import (
    TOOLS,
    CustomerNotFound,
    ProductNotFound,
    check_product_eligibility,
    execute_tool,
    get_customer_profile,
    search_product_catalog,
)

# --- get_customer_profile ----------------------------------------------------------------------


def test_get_customer_profile_returns_minimised_view() -> None:
    view = get_customer_profile("CUST-001")

    assert isinstance(view, ModelCustomerView)
    assert view.customer_ref == "CUST-001"
    assert view.risk_tolerance == "low"
    assert "Anna" not in view.model_dump_json()


def test_get_customer_profile_unknown_customer() -> None:
    with pytest.raises(CustomerNotFound):
        get_customer_profile("CUST-999")


# --- search_product_catalog --------------------------------------------------------------------


def test_search_low_risk_retirement_ranks_low_risk_products_first() -> None:
    results = search_product_catalog("low risk retirement")

    assert {p.product_id for p in results[:2]} == {"PROD-001", "PROD-002"}


def test_search_growth_finds_equity_products() -> None:
    ids = {p.product_id for p in search_product_catalog("high growth equity")}

    assert "PROD-004" in ids


def test_search_respects_max_results() -> None:
    assert len(search_product_catalog("risk bonds equities growth", max_results=2)) <= 2


def test_search_unrelated_query_returns_nothing() -> None:
    assert search_product_catalog("weather in Paris") == []


# --- check_product_eligibility -----------------------------------------------------------------


def test_eligible_product() -> None:
    result = check_product_eligibility("CUST-001", "PROD-001")

    assert result.allowed
    assert result.reasons == []


@pytest.mark.parametrize(
    ("customer_id", "product_id", "rule"),
    [
        ("CUST-001", "PROD-004", "R2_risk_mismatch"),  # low risk customer, high risk product
        ("CUST-001", "PROD-003", "R2_risk_mismatch"),  # low risk customer, medium risk product
        ("CUST-003", "PROD-003", "R3_horizon_mismatch"),  # 2y horizon, 5y product
        ("CUST-005", "PROD-005", "R4_prohibited_product"),  # complex product, age 23
    ],
)
def test_ineligible_products(customer_id: str, product_id: str, rule: str) -> None:
    result = check_product_eligibility(customer_id, product_id)

    assert not result.allowed
    assert any(r.startswith(rule) for r in result.reasons)


def test_high_risk_customer_can_hold_high_risk_product() -> None:
    assert check_product_eligibility("CUST-002", "PROD-004").allowed


def test_incomplete_profile_is_not_eligible() -> None:
    result = check_product_eligibility("CUST-004", "PROD-001")

    assert not result.allowed
    assert "profile incomplete" in result.reasons[0]


def test_eligibility_unknown_product() -> None:
    with pytest.raises(ProductNotFound):
        check_product_eligibility("CUST-001", "PROD-999")


# --- Registry ----------------------------------------------------------------------------------


def test_registry_definitions_have_json_schemas() -> None:
    assert set(TOOLS) == {
        "get_customer_profile",
        "search_product_catalog",
        "check_product_eligibility",
        "retrieve_policy_context",
    }
    for spec in TOOLS.values():
        definition = spec.definition()
        assert definition["input_schema"]["type"] == "object"
        assert definition["description"]


def test_execute_tool_success_records_call(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app.tools"):
        result, call = execute_tool("get_customer_profile", {"customer_id": "CUST-002"})

    assert isinstance(result, ModelCustomerView)
    assert call.success
    assert call.duration_ms >= 0
    assert "CUST-002" in call.result_summary
    assert "tool_call name=get_customer_profile success=True" in caplog.text


def test_execute_tool_summaries_contain_no_names() -> None:
    _, call = execute_tool("get_customer_profile", {"customer_id": "CUST-001"})

    assert "Anna" not in call.result_summary


def test_execute_tool_unknown_tool() -> None:
    result, call = execute_tool("transfer_funds", {"amount": 1_000_000})

    assert result is None
    assert not call.success
    assert "unknown tool" in call.result_summary


def test_execute_tool_invalid_arguments() -> None:
    result, call = execute_tool("get_customer_profile", {"customer_id": "Anna"})

    assert result is None
    assert not call.success
    assert "invalid arguments" in call.result_summary


def test_execute_tool_rejects_extra_arguments() -> None:
    _, call = execute_tool(
        "check_product_eligibility",
        {"customer_id": "CUST-001", "product_id": "PROD-004", "override": True},
    )

    assert not call.success


def test_execute_tool_domain_error_is_reported_not_raised() -> None:
    result, call = execute_tool("get_customer_profile", {"customer_id": "CUST-999"})

    assert result is None
    assert call.result_summary == "error: customer CUST-999 not found"

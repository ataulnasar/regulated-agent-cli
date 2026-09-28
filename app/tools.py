"""Deterministic tools and the registry the agent (and later the MCP server) call them through.

The model never touches data directly. It gets an allowlist of named capabilities, each with a
validated input schema, and every call is timed, logged and recorded as a ToolCall.
"""

import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from app import rag
from app.data import load_customers, load_products
from app.models import (
    CustomerId,
    CustomerProfile,
    EligibilityResult,
    ModelCustomerView,
    PolicySection,
    Product,
    ProductId,
    ToolCall,
)
from app.observability import log_event
from app.rules import missing_profile_fields, suitability_violations

logger = logging.getLogger(__name__)


class DomainError(Exception):
    """An expected failure whose message is safe to show to the model."""


class CustomerNotFound(DomainError):
    pass


class ProductNotFound(DomainError):
    pass


# --- Tools -------------------------------------------------------------------------------------


def load_customer_profile(customer_id: str) -> CustomerProfile:
    """Full profile, for internal use (guardrail). Not exposed to the model."""
    try:
        return load_customers()[customer_id]
    except KeyError:
        raise CustomerNotFound(f"customer {customer_id} not found") from None


def get_product(product_id: str) -> Product:
    try:
        return load_products()[product_id]
    except KeyError:
        raise ProductNotFound(f"product {product_id} not found") from None


def get_customer_profile(customer_id: str) -> ModelCustomerView:
    """The minimised view of a customer that the model is allowed to see."""
    return ModelCustomerView.from_profile(load_customer_profile(customer_id))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def search_product_catalog(query: str, max_results: int = 5) -> list[Product]:
    """Keyword search over the catalogue, weighting rare terms higher (IDF).

    Terms that appear in every product (e.g. "fictional") score zero, so they cannot make every
    product look relevant.
    """
    products = list(load_products().values())
    docs = {
        p.product_id: _tokens(f"{p.name} {p.risk_level} risk {p.description}") for p in products
    }
    query_tokens = _tokens(query)

    def idf(token: str) -> float:
        df = sum(token in doc for doc in docs.values())
        return math.log(len(docs) / df) if df else 0.0

    scored = [(sum(idf(t) for t in query_tokens & docs[p.product_id]), p) for p in products]
    ranked = sorted((item for item in scored if item[0] > 0), key=lambda item: -item[0])
    return [p for _, p in ranked[:max_results]]


def check_product_eligibility(customer_id: str, product_id: str) -> EligibilityResult:
    """ADVISORY check so the model can steer away from unsuitable products.

    The compliance guardrail re-applies the same rules after the model has answered, so skipping or
    ignoring this tool cannot make an unsuitable recommendation pass (defense in depth).
    """
    profile = load_customer_profile(customer_id)
    product = get_product(product_id)

    reasons = [f"{v.rule_id}: {v.reason}" for v in suitability_violations(profile, product)]
    if missing := missing_profile_fields(profile):
        reasons.append(f"profile incomplete ({', '.join(missing)}): gather information first")
    return EligibilityResult(allowed=not reasons, reasons=reasons)


# --- Tool inputs (become the JSON schemas the model sees) --------------------------------------


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetCustomerProfileInput(ToolInput):
    customer_id: CustomerId


class SearchProductCatalogInput(ToolInput):
    query: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    max_results: Annotated[int, Field(ge=1, le=10)] = 5


class CheckProductEligibilityInput(ToolInput):
    customer_id: CustomerId
    product_id: ProductId


class RetrievePolicyContextInput(ToolInput):
    query: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    top_k: Annotated[int, Field(ge=1, le=5)] = 3


# --- Registry ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[ToolInput]
    fn: Callable[..., Any]
    summarise: Callable[[Any], str]

    def definition(self) -> dict[str, Any]:
        """Tool definition in the shape Claude's Messages API (and MCP) expect."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": tool_schema(self.input_model),
        }


def tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for a tool input: $refs inlined, internal docstring and titles removed."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    schema.pop("description", None)
    return resolve(schema)


def _summarise_customer(view: ModelCustomerView) -> str:
    return f"{view.customer_ref} risk={view.risk_tolerance} horizon={view.investment_horizon_years}"


def _summarise_products(products: list[Product]) -> str:
    return f"{len(products)} products: {', '.join(p.product_id for p in products) or 'none'}"


def _summarise_eligibility(result: EligibilityResult) -> str:
    rules = [r.split(":")[0] for r in result.reasons]
    return f"allowed={result.allowed}" + (f" ({', '.join(rules)})" if rules else "")


def _summarise_policies(sections: list[PolicySection]) -> str:
    return f"{len(sections)} sections: {', '.join(s.citation for s in sections) or 'none'}"


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "get_customer_profile",
            "Get the customer's risk tolerance, investment horizon and banded age/income.",
            GetCustomerProfileInput,
            get_customer_profile,
            _summarise_customer,
        ),
        ToolSpec(
            "search_product_catalog",
            "Keyword search of the product catalogue. Only products returned here may be "
            "recommended.",
            SearchProductCatalogInput,
            search_product_catalog,
            _summarise_products,
        ),
        ToolSpec(
            "check_product_eligibility",
            "Check whether a product is suitable for a customer under the suitability rules.",
            CheckProductEligibilityInput,
            check_product_eligibility,
            _summarise_eligibility,
        ),
        ToolSpec(
            "retrieve_policy_context",
            "Retrieve the policy sections relevant to a query, with citation ids.",
            RetrievePolicyContextInput,
            rag.retrieve_policy_context,
            _summarise_policies,
        ),
    )
}


def execute_tool(name: str, raw_args: dict[str, Any]) -> tuple[Any, ToolCall]:
    """Validate, run, time and log one tool call. Never raises into the agent loop.

    Returns (result, ToolCall). On failure result is None and ToolCall.result_summary holds an
    error message that is safe to send back to the model.
    """
    start = time.perf_counter()
    result: Any = None
    try:
        spec = TOOLS[name]
    except KeyError:
        spec, summary = None, f"error: unknown tool '{name}'"
    else:
        try:
            args = spec.input_model.model_validate(raw_args)
            result = spec.fn(**args.model_dump())
            summary = spec.summarise(result)
        except ValidationError as exc:
            summary = f"error: invalid arguments: {_validation_summary(exc)}"
        except DomainError as exc:
            summary = f"error: {exc}"
        except Exception:
            logger.exception("tool %s failed unexpectedly", name)
            summary = "error: internal tool failure"

    success = spec is not None and not summary.startswith("error:")
    call = ToolCall(
        tool_name=name,
        arguments=raw_args,
        success=success,
        result_summary=summary[:300],
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
    )
    # Arguments are left out on purpose: model-written queries can echo the customer's request.
    log_event(
        logger,
        "tool_call",
        tool=call.tool_name,
        success=call.success,
        duration_ms=call.duration_ms,
        summary=call.result_summary,
    )
    return (result if success else None), call


def _validation_summary(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'input'}: {err['msg']}" for err in exc.errors()
    )

"""Loaders for the fictional demo data. Every record is validated on load."""

from functools import cache
from pathlib import Path

from pydantic import TypeAdapter

from app.models import CustomerProfile, Product

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POLICIES_PATH = DATA_DIR / "policies.md"


@cache
def load_customers() -> dict[str, CustomerProfile]:
    raw = (DATA_DIR / "customers.json").read_bytes()
    customers = TypeAdapter(list[CustomerProfile]).validate_json(raw)
    return {c.customer_id: c for c in customers}


@cache
def load_products() -> dict[str, Product]:
    raw = (DATA_DIR / "products.json").read_bytes()
    products = TypeAdapter(list[Product]).validate_json(raw)
    return {p.product_id: p for p in products}


def load_policies_markdown() -> str:
    return POLICIES_PATH.read_text(encoding="utf-8")

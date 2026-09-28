import pytest

from app.data import load_policies_markdown
from app.rag import (
    all_citation_ids,
    parse_policies,
    retrieve_policy_context,
    score_sections,
    tokenize,
)
from app.tools import execute_tool


def ids(query: str, top_k: int = 3) -> list[str]:
    return [s.section_id for s in retrieve_policy_context(query, top_k=top_k)]


def test_parse_policies_uses_stable_anchors() -> None:
    sections = parse_policies(load_policies_markdown())
    by_id = {s.section_id: s for s in sections}

    assert list(by_id) == [
        "suitability",
        "human-review",
        "evidence",
        "incomplete-profile",
        "prohibited",
        "four-eyes",
    ]
    assert by_id["human-review"].citation == "policies.md#human-review"
    assert by_id["human-review"].heading == "Human Review"
    assert "Retirement-related" in by_id["human-review"].text
    assert "##" not in by_id["suitability"].text


@pytest.mark.parametrize(
    ("query", "expected_top"),
    [
        ("retirement", "human-review"),
        ("retirement pension advice for a 58 year old", "human-review"),
        ("risk tolerance", "suitability"),
        ("investment horizon too short for the product", "suitability"),
        ("horizon missing", "incomplete-profile"),
        ("leveraged complex product", "prohibited"),
        ("guaranteed returns", "prohibited"),
        ("who can approve, same advisor", "four-eyes"),
        ("citations and sources", "evidence"),
    ],
)
def test_query_retrieves_expected_section_first(query: str, expected_top: str) -> None:
    assert ids(query)[0] == expected_top


@pytest.mark.parametrize("query", ["weather in Paris", "best pizza in Stockholm", "the and of"])
def test_unrelated_queries_return_nothing(query: str) -> None:
    assert retrieve_policy_context(query) == []


def test_top_k_limits_results() -> None:
    assert len(retrieve_policy_context("risk product customer horizon advisor", top_k=2)) == 2


def test_scores_are_sorted_and_deterministic() -> None:
    first = score_sections("retirement risk")
    second = score_sections("retirement risk")

    assert first == second
    scores = [score for score, _ in first]
    assert scores == sorted(scores, reverse=True)


def test_tokenize_folds_plurals_and_drops_stopwords() -> None:
    assert tokenize("The products must not exceed") == ["product", "exceed"]


def test_all_citation_ids_cover_policies_and_products() -> None:
    citations = all_citation_ids()

    assert "policies.md#suitability" in citations
    assert "products.json#PROD-001" in citations
    assert "policies.md#made-up" not in citations


def test_registered_tool_returns_citations() -> None:
    result, call = execute_tool("retrieve_policy_context", {"query": "retirement"})

    assert call.success
    assert result[0].citation == "policies.md#human-review"
    assert "policies.md#human-review" in call.result_summary

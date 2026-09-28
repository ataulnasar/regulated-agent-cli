"""Lightweight retrieval over data/policies.md, for grounding and citations.

Deliberately lexical (BM25) rather than a vector database:
- the corpus is a handful of short sections, so embeddings add infrastructure without adding recall;
- lexical scores are deterministic and explainable ("matched 'retirement' in human-review");
- retrieve_policy_context() is the only interface, so an embedding or hybrid retriever can replace
  it without touching the agent or the guardrail.

Products are not indexed here: search_product_catalog already returns them with their source_id,
and the agent harness records those as retrieved citations too.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache

from app.data import load_policies_markdown, load_products
from app.models import PolicySection

POLICY_FILE = "policies.md"
HEADING = re.compile(r"^## (?P<heading>.+?) \{#(?P<anchor>[a-z0-9-]+)\}\s*$", re.MULTILINE)
STOPWORDS = frozenset(
    "a an and any are as be before by do for from if in is it must not of on or such that the "
    "this to was were with".split()
)
K1, B = 1.5, 0.75


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    # Minimal stemming: fold plurals so "products" matches "product".
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokens if t not in STOPWORDS]


def parse_policies(markdown: str) -> list[PolicySection]:
    """Split on '## Heading {#anchor}'. Anything before the first section is ignored."""
    matches = list(HEADING.finditer(markdown))
    sections = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        anchor = match["anchor"]
        sections.append(
            PolicySection(
                section_id=anchor,
                heading=match["heading"],
                text=" ".join(markdown[match.end() : end].split()),
                citation=f"{POLICY_FILE}#{anchor}",
            )
        )
    return sections


@dataclass(frozen=True)
class _Index:
    sections: list[PolicySection]
    term_freqs: list[Counter[str]]
    doc_lengths: list[int]
    avg_length: float
    doc_freq: Counter[str]


@cache
def _index() -> _Index:
    sections = parse_policies(load_policies_markdown())
    # The heading is counted twice: it is the strongest signal of what a section is about.
    term_freqs = [Counter(tokenize(f"{s.heading} {s.heading} {s.text}")) for s in sections]
    lengths = [sum(tf.values()) for tf in term_freqs]
    return _Index(
        sections=sections,
        term_freqs=term_freqs,
        doc_lengths=lengths,
        avg_length=sum(lengths) / len(lengths),
        doc_freq=Counter(term for tf in term_freqs for term in tf),
    )


def score_sections(query: str) -> list[tuple[float, PolicySection]]:
    """BM25 score for every section, highest first. Useful for debugging retrieval."""
    index = _index()
    n = len(index.sections)
    query_terms = set(tokenize(query))
    scored = []
    docs = zip(index.sections, index.term_freqs, index.doc_lengths, strict=True)
    for section, tf, length in docs:
        score = 0.0
        for term in query_terms & tf.keys():
            df = index.doc_freq[term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            norm = tf[term] + K1 * (1 - B + B * length / index.avg_length)
            score += idf * tf[term] * (K1 + 1) / norm
        scored.append((round(score, 3), section))
    return sorted(scored, key=lambda item: -item[0])


def retrieve_policy_context(
    query: str, top_k: int = 3, min_score: float = 0.5
) -> list[PolicySection]:
    """The most relevant policy sections for a query. Weak matches are dropped, not padded."""
    return [section for score, section in score_sections(query)[:top_k] if score >= min_score]


def all_citation_ids() -> set[str]:
    """Every citation that exists, so the guardrail can reject invented ones."""
    policy = {s.citation for s in _index().sections}
    products = {p.source_id for p in load_products().values()}
    return policy | products

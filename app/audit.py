"""Append-only, hash-chained audit trail.

Each record stores the hash of the previous record, and its own hash covers its content plus that
previous hash. Editing, deleting or reordering any line breaks the chain from that point on, and
`verify-audit` reports where.

What is recorded: who decided what, when, under which rules and evidence, with which model. What is
not: the customer's name, income or the request text (only its SHA-256 fingerprint).

Production would put this in WORM storage or an append-only table with the chain head anchored
externally. A local JSONL file shows the mechanism.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.data import DATA_DIR
from app.models import FinalAssessment
from app.observability import fingerprint

AUDIT_PATH = DATA_DIR / "audit-log.jsonl"
GENESIS_HASH = "0" * 64


def _canonical(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(prev_hash: str, record: dict[str, Any]) -> str:
    return hashlib.sha256((prev_hash + _canonical(record)).encode("utf-8")).hexdigest()


def _last_hash(path: Path) -> str:
    if not path.is_file():
        return GENESIS_HASH
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1])["hash"] if lines else GENESIS_HASH


def append_event(event: str, assessment: FinalAssessment, actor: str) -> dict[str, Any]:
    rec, compliance = assessment.recommendation, assessment.compliance
    record: dict[str, Any] = {
        "event": event,
        "timestamp": datetime.now(UTC).isoformat(),
        "actor": actor,
        "assessment_id": str(assessment.assessment_id),
        "trace_id": assessment.trace_id,
        "customer_id": assessment.customer_id,
        "advisor_id": assessment.advisor_id,
        "model": assessment.model,
        "request_sha256": fingerprint(assessment.request),
        "intent": rec.intent,
        "recommended_action": rec.recommended_action,
        "recommended_product_id": rec.recommended_product_id,
        "citations": rec.citations,
        "allowed": compliance.allowed,
        "requires_human_review": compliance.requires_human_review,
        "rules_triggered": compliance.rules_triggered,
        "status": assessment.status,
        "tools_called": [c.tool_name for c in assessment.tool_calls],
    }
    if assessment.metrics:
        record["metrics"] = assessment.metrics.model_dump()

    path = AUDIT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    record["prev_hash"] = _last_hash(path)
    record["hash"] = _hash(record["prev_hash"], record)
    with path.open("a", encoding="utf-8") as f:
        f.write(_canonical(record) + "\n")
    return record


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    records: int
    first_bad_line: int | None = None
    problem: str | None = None


def verify_chain(path: Path | None = None) -> VerifyResult:
    path = path or AUDIT_PATH
    if not path.is_file():
        return VerifyResult(ok=True, records=0)
    prev_hash = GENESIS_HASH
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
            stored_hash = record.pop("hash")
        except (json.JSONDecodeError, KeyError):
            return VerifyResult(False, number - 1, number, "unreadable record")
        if record.get("prev_hash") != prev_hash:
            return VerifyResult(False, number - 1, number, "chain broken (record missing or moved)")
        if _hash(prev_hash, record) != stored_hash:
            return VerifyResult(False, number - 1, number, "content altered")
        prev_hash = stored_hash
    return VerifyResult(ok=True, records=len(lines))

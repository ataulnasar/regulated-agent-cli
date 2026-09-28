"""Structured JSON logging with a per-assessment trace ID, and redaction of personal data.

Logs carry IDs, counts, timings and outcomes. They never carry names, income or the raw request
text. As a second line of defence, every log line passes through redact() before it is written.
"""

import hashlib
import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

# Personal data that could arrive through free text (the request, or model-written tool arguments).
REDACTIONS = [
    (re.compile(r"\b(?:19|20)?\d{6}[-+]?\d{4}\b"), "[PERSONNUMMER]"),  # Swedish personal ID number
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),
    (re.compile(r"(?<!\w)\+?\d[\d -]{7,}\d\b"), "[PHONE]"),
]


def redact(text: str) -> str:
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def fingerprint(text: str) -> str:
    """A stable reference to sensitive text without storing the text itself."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    # The trace ID is captured when the event happens, not when the line is formatted.
    if trace_id := trace_id_var.get():
        fields = {"trace_id": trace_id, **fields}
    logger.log(level, event, extra={"fields": fields})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        if trace_id := trace_id_var.get():
            payload["trace_id"] = trace_id
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, default=str, ensure_ascii=False))


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever sys.stderr is at emit time (so redirection and test capture work)."""

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, _value) -> None:
        pass


def setup_logging(level: str | int = logging.WARNING) -> None:
    """JSON logs for the `app` package on stderr, keeping stdout free for the demo output."""
    logger = logging.getLogger("app")
    logger.setLevel(level)
    if not any(isinstance(h, _StderrHandler) for h in logger.handlers):
        handler = _StderrHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

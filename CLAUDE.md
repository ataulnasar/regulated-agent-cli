# regulated-agent-cli

A small, runnable reference implementation of an agentic AI workflow for a regulated
wealth-management business (Nordic fintech context).

## Who uses it
A financial advisor / sales rep, NOT the end customer. The agent drafts a product recommendation
for a customer; deterministic compliance rules decide whether it is allowed and whether a second
human must approve it (MiFID II-style suitability assessment).

## Principles
- Controlled autonomy: the model chooses among allowlisted tools, but it never owns compliance
  decisions. Deterministic code has the final say.
- Everything the model produces is validated through Pydantic. On invalid output, fail safe
  (escalate to human review); never guess.
- It must run with no API key (a fake provider) so a reviewer can clone it, run it, and run the evals.
- Minimal data goes to the model. No personal data in logs.
- Python 3.12, Typer, Pydantic v2, pytest, Rich. Official `anthropic` SDK. No LangChain or other
  agent frameworks.
- Keep the code small and readable during a live walkthrough. Prefer clarity over abstraction.

## Commands
- Install: `uv sync`
- Run: `uv run python -m app.cli assess --customer CUST-001 --advisor ADV-01 --request "..."`
- Test: `uv run pytest`
- Lint: `uv run ruff check .`

# Regulated Agent CLI

An advisor-facing suitability assistant built to show controlled autonomy: the model plans and
drafts, deterministic code decides, and a human approves where regulation requires it.

> Work in progress. The full README (architecture, design decisions, evals) follows once the
> core workflow is in place.

All customer and product data in this repository is fictional. Nothing here is financial advice.

## Quickstart

```bash
uv sync
uv run python -m app.cli assess --customer CUST-001 --advisor ADV-01 \
  --request "I want a low-risk investment for retirement"
uv run pytest
```

No API key is needed; the default provider is an offline fake.

# Regulated Agent CLI

A small, runnable reference implementation of an agentic AI workflow for a regulated
wealth-management business. An advisor asks for a product recommendation for a customer; a Claude
agent gathers evidence through allowlisted tools and drafts one; deterministic compliance rules
decide whether it may be released, rejected, or needs a second advisor's approval. Every outcome
lands in a tamper-evident audit trail.

The goal is not maximum autonomy. It is **controlled autonomy**: the model plans and drafts, code
decides, and humans approve where regulation requires it.

> All customer, product and policy data is fictional. Nothing here is financial advice.

## Quickstart

No API key needed. The default provider is an offline, deterministic fake.

```bash
uv sync --all-extras
uv run python -m app.cli assess --customer CUST-001 --advisor ADV-01 \
  --request "I want a low-risk investment for retirement"
uv run python -m app.cli review <assessment-id> --reviewer ADV-02    # second advisor decides
uv run python -m app.cli verify-audit                                # check the audit hash chain
uv run python -m evals.run_evals                                     # eval suite
uv run pytest                                                        # 199 tests
```

With a real model:

```bash
export ANTHROPIC_API_KEY=...        # or: ant auth login
uv run python -m app.cli assess --provider anthropic --customer CUST-002 --advisor ADV-01 \
  --request "I want long-term growth"
uv run python -m evals.run_evals --provider anthropic      # 3 repeats per case

# Amazon Bedrock, Stockholm region by default (AWS credentials from the standard chain)
uv run python -m app.cli assess --provider bedrock ...
```

`-v` writes structured JSON logs to stderr. Model and region are set with `MODEL_ID`,
`BEDROCK_MODEL_ID` and `AWS_REGION` (see `.env.example`).

## Architecture

```mermaid
flowchart TD
    A["Advisor (CLI or MCP client)"] -->|"customer id + request"| L
    P["Model provider: Anthropic API / Bedrock eu-north-1 / offline fake"] <--> L
    subgraph Harness
        L["Bounded tool-use loop (max 8 steps)"] <-->|"allowlisted calls"| T["Tool registry (validated inputs)"]
        T --> C["get_customer_profile (minimised view)"]
        T --> S["search_product_catalog"]
        T --> E["check_product_eligibility (advisory)"]
        T --> R["retrieve_policy_context (BM25 over policies.md)"]
        L -->|"submit_recommendation (schema-validated)"| G
    end
    G["Deterministic guardrail: R1-R4 reject, H1-H6 review"] -->|"auto_approved / rejected / declined"| O["Outcome"]
    G -->|"pending_review"| H["Second advisor (four-eyes)"]
    H --> O
    O --> AU[("Hash-chained audit log")]
```

| Module | Responsibility |
|---|---|
| `app/agent.py` | The tool-use loop: allowlist, citation tracking, validation, fallback, hand-off to the guardrail |
| `app/tools.py` | Tool registry and the four deterministic tools |
| `app/rag.py` | BM25 retrieval over the policy document, with stable citations |
| `app/rules.py`, `app/guardrails.py` | Suitability rules and the compliance guardrail |
| `app/providers/` | Model abstraction: Anthropic, Bedrock, offline fake |
| `app/review.py` | Pending queue and four-eyes human review |
| `app/audit.py`, `app/observability.py` | Hash-chained audit trail, JSON logs, redaction |
| `app/mcp_server.py` | Optional MCP server over the same registry |
| `evals/` | Behaviour and harness eval cases, runner, thresholds |

## A sample run

```
─────────────────── REGULATED AGENT  trace 2cec0fe6063f  ·  fake/fake-heuristic ───────────────────
╭─ Agent ──────────────────────────────────────────────────────────────────────────────────────╮
│ Intent: retirement_planning                                                                  │
│  ✓  get_customer_profile       CUST-001 risk=low horizon=10                                  │
│  ✓  search_product_catalog     2 products: PROD-001, PROD-002                                │
│  ✓  retrieve_policy_context    3 sections: policies.md#suitability, policies.md#human-review │
│  ✓  check_product_eligibility  allowed=True                                                  │
│  ✓  submit_recommendation      accepted: recommend_product PROD-001                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Recommendation ─────────────────────────────────────────────────────────────────────────────╮
│ Product           Nordic Stable Fund (PROD-001)                                              │
│ Model confidence  80% (self-reported, not used for decisions)                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Compliance ─────────────────────────────────────────────────────────────────────────────────╮
│ ⚠ HUMAN REVIEW REQUIRED                                                                      │
│  H1_retirement_intent  retirement recommendations need a second advisor's approval           │
│                        (policies.md#human-review)                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
NOT RELEASED — must not be presented to the customer.
```

## How a request flows

1. The CLI (or MCP client) calls `run_assessment(customer, advisor, request, provider)`. Unknown
   customers and invalid advisor IDs fail before any model call.
2. The model gets a system prompt, the request fenced in `<customer_request>`, and five tools: the
   four business tools and `submit_recommendation`, whose input schema *is* the `AgentRecommendation`
   model.
3. The loop runs at most 8 steps. Tools execute only through the registry; unknown tool names and
   invalid arguments come back to the model as errors, never exceptions. Tool output is fenced in
   `<tool_output>`.
4. The harness records which citations were **actually returned by tools**. The model's claims about
   its evidence are checked against that set, never trusted.
5. The submission is validated. One correction is allowed; after that, or if the model stops or
   runs out of steps, a fallback sends the case to human review.
6. The guardrail evaluates the recommendation against the **full** customer profile and sets the
   status. The model never sets status.
7. `pending_review` cases wait for a second advisor. Every creation and decision is audited.

## Guardrail rules

| Rule | Outcome | Policy |
|---|---|---|
| R1 product not in catalogue (hallucination) | reject | |
| R2 product risk exceeds customer risk tolerance | reject | `#suitability` |
| R3 product horizon exceeds customer horizon | reject | `#suitability` |
| R4 complex product to customer under 25 or without high risk tolerance | reject | `#prohibited` |
| H1 retirement intent | human review | `#human-review` |
| H2 no citations, or a citation that was never retrieved | human review | `#evidence` |
| H3 incomplete customer profile | human review | `#incomplete-profile` |
| H4 unclear intent | human review | |
| H5 invalid model output (fallback used, or self-contradictory) | human review | |
| H6 the model itself asked for review | human review | |

A rejection is final and never also asks for review. The model can make the outcome more cautious
(H6), never less. All rules are in `app/guardrails.py` and `app/rules.py`, readable top to bottom.

## Evaluation

`evals/cases.json` holds 13 cases of two kinds:

- **Behaviour** (7): the whole agent runs, offline or against a real model. They cover a clean
  approval, retirement review, a short horizon, an incomplete profile, an ambiguous request, and
  direct and indirect prompt injection. Each case lists the *acceptable* outcomes, because a good
  model may reasonably choose between several safe answers.
- **Harness** (6): a scripted fake plays a misbehaving model: a hallucinated product, an unretrieved
  citation, no citations, a compromised model obeying an injection, `"approved": true`, and skipped
  eligibility checks. These prove the guardrails hold whatever the model does. Fake provider only.

Everything is judged on structured fields, never text. The headline metric is **unsafe approvals**:
a recommendation released to the customer that breaks a suitability rule, rests on an incomplete
profile or a non-existent product, or skipped a review the case required. It must be 0 for every
provider. Real models also need at least 90% case pass rate and status accuracy
(`evals/thresholds.json`), with 3 repeats per case to expose variance. CI runs the offline suite on
every push.

```
runs 13 · case_pass_rate 100% · status_accuracy 100% · intent_accuracy 100% · rule_accuracy 100%
citation_validity 100% · escalation_recall 100% · unsafe_approvals 0 · PASS
```

Offline results measure the harness and guardrails, not model quality. The Anthropic and Bedrock
providers are implemented and unit-tested for configuration, but the numbers above are from the fake;
`--provider anthropic` produces the real-model baseline.

## Design decisions

**One bounded agent, not a multi-agent system.** The task is a single, short, auditable decision.
More agents would add hand-offs, latency, cost and places for context to get lost, without adding a
capability. *In production:* the natural split is an independent compliance-review agent with its own
context critiquing the draft. The deterministic guardrail would still have the final say.

**Native tool use with a schema-validated final submission.** The model finishes by calling
`submit_recommendation`, whose input schema is generated from the Pydantic model, with extra fields
forbidden. No free-text parsing. The loop is hand-written rather than the SDK's beta tool runner
because it needs custom validation, a correction round and citation tracking.

**The guardrail is the authority; the eligibility tool is advisory.** The same rule functions serve
both. The tool helps the model avoid unsuitable products; the guardrail re-applies the rules after the
model has answered. The model can skip or ignore the tool; it cannot skip the guardrail.

**No gating on model confidence.** It is self-reported and uncalibrated, so a threshold on it would let
the model talk its way past review. It is displayed and logged, never used. Review triggers are
things code can verify.

**Lexical RAG, not a vector database.** Six short policy sections. BM25 in 20 lines of plain Python
is deterministic, explainable ("matched *retirement* in human-review") and needs no infrastructure.
`retrieve_policy_context()` is the only interface, so hybrid or embedding retrieval can replace it.
*In production:* chunked policy corpora in a managed vector store with hybrid search, and citation
checks unchanged.

**Data minimisation.** The model sees a `ModelCustomerView`: no name, banded age and income. Logs and
the audit trail hold IDs, rules and outcomes, never names, income or request text (a SHA-256
fingerprint stands in). `redact()` masks personal ID numbers (personnummer), emails, cards and phone
numbers on every log line as a second layer. Bedrock defaults to `eu-north-1` for EU data residency.

**Four-eyes review and a hash-chained audit trail.** The requesting advisor cannot approve their own
case. Each audit record chains the previous record's hash, so edits, deletions and reordering are
detectable (`verify-audit`). *In production:* WORM storage or an append-only table with the chain head
anchored externally, and reviewer identity from IAM/SSO instead of a CLI flag.

**Prompt injection: authority separation, not prompt hardening.** Requests and tool output are fenced
and fake closing tags are neutralised, which reduces the chance the model follows injected text. But
the defence does not depend on it: the model has no field to approve with, and the guardrail decides.
The tests use a model that deliberately obeys the injection and show nothing unsuitable is released.
`PROD-005`'s description contains a planted "SYSTEM NOTE" for exactly this.

**Offline-first.** A deterministic fake provider makes the demo, tests and evals run without keys or
network, including in CI. It has a scripted mode (to simulate specific misbehaviour) and a heuristic
mode (a well-behaved reference agent).

## Security and governance

- **Access:** tools are read-only; the only state changes are the pending queue and the audit trail.
  The MCP server deliberately exposes no approval tool.
- **Inputs:** every tool input, model submission, reviewer ID and data file is validated by Pydantic
  with extra fields forbidden.
- **Traceability:** every assessment has a trace ID across logs, the assessment record and the audit
  trail, which also records the model, rules, citations, tokens and latency.
- **Out of scope for the demo:** authentication, encryption at rest, retention policies, and
  concurrent writers to the audit log. The pending queue (`data/pending/`) stores the full request
  because the reviewer needs it; the minimised records are the logs and audit trail.

## Path to production

- FastAPI service with a small advisor UI; the CLI becomes one client among several.
- Postgres for assessments, the review queue and an append-only audit table; reviewer identity and
  roles from the firm's IdP.
- Claude on Bedrock in `eu-north-1`, OpenTelemetry traces, and cost and latency budgets per request.
- The eval suite grows from real advisor feedback: every disputed recommendation becomes a case.
- Policy rules owned by compliance as reviewed data, with the guardrail tested against each change.

## MCP server (optional)

The same capabilities are available to any MCP client (Claude Code, Claude Desktop):

```bash
uv run --extra mcp python -m app.mcp_server   # stdio transport
```

| Tool | What it does |
|---|---|
| `get_customer_profile` | Minimised customer view (no name, banded age and income) |
| `search_product_catalog` | Keyword search of the product catalogue |
| `check_product_eligibility` | Suitability rules for one customer and product |
| `retrieve_policy_context` | Relevant policy sections with citation ids |
| `assess_request` | The full agent, guardrail and audit trail; queues pending reviews |

The server reuses the tool registry, so schemas, validation, logging and the audit trail are
identical to the CLI. There is deliberately **no approval tool**: approval stays with an identified
second advisor through `python -m app.cli review`.

**Claude Code** offers to enable the project's `.mcp.json` server when you open this repository.
**Claude Desktop**: add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "regulated-agent": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/regulated-agent-cli",
               "run", "--extra", "mcp", "python", "-m", "app.mcp_server"],
      "env": {"LLM_PROVIDER": "fake"}
    }
  }
}
```

`skills/suitability-assessment/SKILL.md` is a Claude Skill that tells Claude how to use these tools:
when to call `assess_request`, how to report each status, and what it must never do.

## Demo data

| Customer | Profile | Demonstrates |
|---|---|---|
| CUST-001 | 58, low risk, 10 years | Retirement case, needs a second advisor's approval |
| CUST-002 | 34, high risk, 20 years | A clean approval path |
| CUST-003 | 45, medium risk, 2 years | Horizon too short for most products |
| CUST-004 | 40, medium risk, horizon missing | Incomplete profile, must escalate |
| CUST-005 | 23, high risk, 15 years | Complex product blocked despite high risk tolerance |

`data/policies.md` holds the rules in plain language, with a stable anchor per section so every
citation can be verified mechanically.

## How I built it

In small, daily-closeable steps, one commit each: typed domain models and fictional data first, then
the deterministic tools, retrieval and guardrail with their tests, before any model was connected.
Then the provider abstraction, the agent loop, human review, the audit trail, the eval suite,
injection hardening and the MCP server. I used a coding agent (Claude Code) with one scoped prompt per
step, and reviewed and tested each step before moving on.

# Working agreement

**Read `ARCHITECTURE.md` before doing anything.** It is the single source of truth for
this repo: product scope, domain model, component design, repo layout, dependency
groups, and the M0–M8 milestone plan with acceptance criteria.

## Project

**Argus** — a multi-tenant corporate expense management platform. Tracks expenses in
real time, enforces policy limits, forecasts spend, detects fraud, and exposes an
agentic copilot. Built as a learning vehicle for production AI/ML engineering:
LangGraph durable workflows, permission-aware RAG, MLOps with drift and retraining,
guardrails, fine-grained authz, and full observability.

## How we work

- **Milestone order is not optional.** Complete M0 before M1, and so on. Each milestone
  has acceptance criteria in `ARCHITECTURE.md` §17 — meet them before moving on.
- **Plan before you build.** At the start of a milestone, produce a task breakdown and
  wait for confirmation before writing code.
- **Small commits.** Conventional Commits format. Commit at logical boundaries, not once
  per milestone.
- **Tests are part of the work,** not a follow-up. Nothing is done until `make test` and
  `make lint` pass.
- **Explain as you go.** I am using this project to prepare for AI Engineer and MLOps
  Engineer interviews, so when you make a non-obvious choice, say why in one or two
  sentences. If the reasoning is durable, put it in an ADR instead.
- **Ask when genuinely ambiguous.** If a decision has real trade-offs and the doc does
  not settle it, ask rather than guessing. If it is minor, decide, write an ADR, and
  move on.

## Hard rules

- **No secrets** in code, env files, logs, or commits. Key Vault + managed identity in
  cloud, gitignored `.env` locally.
- **No floats for money.** `Money` value object, minor units, explicit currency.
- **Authorisation filters at query time**, never post-retrieval discard. See §10.
- **Fail closed.** If OpenFGA, OPA, or a guardrail is unavailable, deny. Never fall back
  to permitting an action or auto-approving an expense.
- **No `user_id`, `expense_id`, `session_id`, or raw text as Prometheus labels.** See §13.
- **Every policy decision the agent makes must carry a citation.** No uncited verdicts.
- **`ml/` and `simulator/` never import from `services/`.** `packages/` imports nothing
  from either. Enforced by import-linter in CI.
- **Do not install the `gpu` dependency group into any service image.**

## Commands

```bash
make up          # full local stack
make up-lite     # postgres + redpanda + phoenix only (fast inner loop)
make seed        # load synthetic tenants, org trees, policy corpus
make test        # unit + integration
make lint        # ruff, mypy --strict, import-linter
make eval        # run the eval suite locally
make simulate    # run the traffic/adversarial simulator
make down
```

## Current status

Repo is empty. **Start at M0.** Update `docs/milestones.md` as you go.

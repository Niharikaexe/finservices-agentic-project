# Argus — Corporate Expense Intelligence Platform

Multi-tenant corporate expense management: real-time tracking, policy limit
enforcement, spend forecasting, and fraud detection, with an agentic copilot on top.

Built as a production-grade reference implementation covering LangGraph durable
workflows with HITL, permission-aware RAG, end-to-end MLOps with drift detection and
automated retraining, guardrails, fine-grained authorisation, and full observability —
exercised against a synthetic-world simulator rather than live customers.

## Start here

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** — full design and the M0–M8 build plan
- **[CLAUDE.md](./CLAUDE.md)** — working agreement
- **[docs/pairing-guide.md](./docs/pairing-guide.md)** — how we split the work
- **[docs/milestones.md](./docs/milestones.md)** — current status and the open task

## Quickstart

```bash
uv sync                 # one venv, every workspace member editable
make test               # unit tests
make lint               # ruff + mypy --strict + import-linter
make simulate-org       # generate tenants, org trees and vendors
make up                 # full local stack (see ARCHITECTURE.md §7)
```

`make simulate` (spend included) fails until `fsa_sim.world.spend` is implemented —
that is the current open task.

## Repository map

```
packages/     shared kernel: fsa_common (Money, settings, errors, logging),
              fsa_authz (Principal), fsa_guardrails (UntrustedText), fsa_telemetry
simulator/    the synthetic world — tenants, org trees, vendors, spend, adversaries
ml/           feature layer with point-in-time correctness; fraud + forecast models
services/     expense-api, copilot, ingestion, mcp-tools, fraud-scorer, forecaster
evals/        golden datasets and the CI release gate
infra/        compose configs today; Terraform, k8s and dashboards later
docs/adr/     one ADR per real decision
```

## Status

**M0 — Foundation: in progress.** Workspace, tooling, CI, local stack and the
simulator's org/vendor generators are in. Spend generation is the open task.

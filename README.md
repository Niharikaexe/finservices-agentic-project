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
make simulate           # generate the synthetic world (~15s, 82k claims)
make features           # assemble the 47-feature training matrix
make install-ml         # add lightgbm, shap, optuna, mlflow
make train-fraud        # train + evaluate the fraud model
make up                 # full local stack (see ARCHITECTURE.md §7)
```

`make train-fraud` fails until `fsa_ml.fraud.train.train` is implemented — that is the
current open task. See [docs/notebooks.md](./docs/notebooks.md) for the notebook/Colab
policy.

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

**M0 — Foundation: in progress.** Workspace, tooling, CI and local stack are in. The
simulator generates a full labelled world (82,563 claims, 1,949 fraud positives across
six typologies, 824 planted injection payloads), and the ML feature layer assembles 47
point-in-time-correct features from it. Fraud model training is the open task.

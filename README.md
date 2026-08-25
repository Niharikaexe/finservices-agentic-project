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

Python 3.12+ is the only prerequisite. No `make`, no `uv`, no docker.

```bash
python3 bootstrap.py            # creates .venv and installs everything
source .venv/bin/activate       # Windows: .venv\Scripts\activate

python tasks.py serve           # then open http://localhost:8080
python tasks.py                 # list every task
```

Ask a Sales employee and an Engineering employee the same question. Sales holds an
addendum raising the client-entertainment limit to ₹15,000; Engineering does not and
sees the global ₹5,000. Neither can retrieve the other's document — the permitted set
is a query predicate, not a filter applied to results.

```bash
python tasks.py test            # unit tests
python tasks.py lint            # ruff + mypy --strict + import-linter
python tasks.py acl             # the ACL isolation suite
python tasks.py traffic         # drive persona traffic at a running service
python tasks.py trace           # read the interaction log back, run by run
python tasks.py simulate        # generate the synthetic world (~15s, 82k claims)
python tasks.py up              # full local stack (needs docker; ARCHITECTURE.md §7)
```

`uv` and `make` both still work if you have them — `tasks.py` detects `uv` and uses it,
and the Makefile is unchanged. **[docs/running-locally.md](./docs/running-locally.md)**
covers the run and the logging in full.

`python tasks.py train-fraud` fails until `fsa_ml.fraud.train.train` is implemented —
that is a parked task. See [docs/notebooks.md](./docs/notebooks.md) for the
notebook/Colab policy.

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

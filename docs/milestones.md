# Milestone status

Acceptance criteria live in [ARCHITECTURE.md §17](../ARCHITECTURE.md#17-milestones).
How we split the work: [pairing-guide.md](./pairing-guide.md).

| # | Milestone | Status | Notes |
|---|---|---|---|
| M0 | Foundation | 🟡 In progress | Skeleton, tooling, CI and simulator scaffolding landed |
| M1 | Ledger + authz + permission-aware RAG | ⬜ Not started | |
| M2 | The agent | ⬜ Not started | |
| M2.5 | Observability | ⬜ Not started | |
| M3 | Guardrails + adversarial | ⬜ Not started | |
| M4 | Fraud model | ⬜ Not started | |
| M5 | Forecasting + drift + retrain | ⬜ Not started | |
| M6 | Release gates + game days | ⬜ Not started | |
| M7 | Azure + GPU | ⬜ Not started | |
| M8 | Semantic Kernel comparison + writeup | ⬜ Not started | |

---

## M0 — Foundation: task breakdown

Deviation from the doc, noted deliberately: we are doing the **simulator early**,
before the ledger. `ARCHITECTURE.md` §17 puts the simulator in M3/M4, but every
downstream component consumes its data — the ML platform cannot be built at all
without it, and the ACL suite in M1 needs 200 synthetic users across 2 tenants. Doing
it first means M1 onwards has real data to work against instead of fixtures.

| # | Task | Owner | Status |
|---|---|---|---|
| 0.1 | uv workspace, dependency groups, ruff/mypy/import-linter | Claude | ✅ done |
| 0.2 | `docker-compose.yml` for the full stack + otel/prom/alertmanager configs | Claude | ✅ done |
| 0.3 | `fsa_common`: `Money`, settings, structured logging, error taxonomy | Claude | ✅ done |
| 0.4 | `fsa_telemetry`: the full metric registry from §13 | Claude | ✅ done |
| 0.5 | `fsa_guardrails.UntrustedText` taint type | Claude | ✅ done |
| 0.6 | `.github/workflows/ci.yml`: ruff, mypy, import-linter, pytest | Claude | ✅ done |
| 0.7 | ADRs 0001–0004 | Claude | ✅ done |
| 0.8 | Simulator: entities, config, org tree, vendor catalogue | Claude | ✅ done |
| 0.9 | Simulator: baseline spend generation (`fsa_sim.world.spend`) | Claude | ✅ done |
| 0.10 | Simulator: fraud typologies + injection payloads (`fsa_sim.adversarial`) | Claude | ✅ done |
| **0.11** | **ML training pipeline: features, training, calibration, eval** | **Claude** | 🟡 **open** |
| 0.12 | Alembic baseline migration + RLS policies | Claude | ⬜ blocked on M1 |
| 0.13 | `scripts/seed_db.py` — world → Postgres | Claude | ⬜ blocked on M1 |

### Generated dataset (seed 42)

`make simulate` produces, in about 14 seconds:

| | |
|---|---|
| tenants / departments / users | 3 / 25 / 1,045 |
| vendors (incl. 12 ghost vendors) | 586 |
| expenses | 82,563 |
| fraud positives | 1,949 (2.4%) |
| ...of which confirmed by audit | 1,650 — the rest are **never** confirmed |
| injection payloads planted | 824 |

The gap between *fraud positives* and *confirmed by audit* is the delayed-ground-truth
constraint made concrete: 299 genuinely fraudulent claims are never confirmed, and
`label_as_of` must drop them rather than count them as clean.

### Open task 0.11 — the ML training pipeline

Build the fraud model end to end on top of the generated world:
feature assembly with point-in-time correctness, a LightGBM baseline, calibration,
PR-AUC / recall@k evaluation, SHAP reasons, and the deliberate-leakage experiment.

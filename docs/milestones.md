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
| **0.9** | **Simulator: baseline spend generation** (`fsa_sim.world.spend`) | **You** | 🟡 **open** |
| 0.10 | Simulator: fraud typologies (`fsa_sim.adversarial`) | You | ⬜ next |
| 0.11 | `ml.features.employee_history` + the causality test | You | ⬜ next |
| 0.12 | Alembic baseline migration + RLS policies | Claude | ⬜ blocked on M1 |
| 0.13 | `scripts/seed_db.py` — world → Postgres | Claude | ⬜ blocked on M1 |

### Open task 0.9

Implement `sample_amount_minor` and `generate_baseline_expenses` in
`simulator/src/fsa_sim/world/spend.py`.

The contract is `tests/simulator/test_spend.py` — 16 tests, currently skipped, which
start running as soon as the functions stop raising `NotImplementedError`.

```bash
uv run pytest tests/simulator/test_spend.py -v
```

Done when: those 16 pass, `make lint` is clean, and `make simulate` writes a world
with a plausible expense count.

# Argus — developer entry points. Everything runs locally; cloud is for the
# deployment story, not the inner loop (ARCHITECTURE.md §7).
.DEFAULT_GOAL := help
SHELL := /bin/bash
COMPOSE := docker compose
UV := uv

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── environment ────────────────────────────────────────────────────────────
.PHONY: install
install:  ## Sync the uv workspace (all members editable, dev group included)
	$(UV) sync

.PHONY: install-ml
install-ml:  ## Sync plus the heavy ML group (lightgbm, mlflow, optuna, shap…)
	$(UV) sync --group ml

# ── local stack ────────────────────────────────────────────────────────────
.PHONY: up
up:  ## Full local stack (§7)
	$(COMPOSE) up -d
	@echo "phoenix :6006  grafana :3000  mlflow :5000  prometheus :9090"

.PHONY: up-lite
up-lite:  ## Postgres + Redpanda + Phoenix only — the fast inner loop
	$(COMPOSE) up -d postgres redpanda phoenix

.PHONY: down
down:  ## Stop the stack (volumes preserved)
	$(COMPOSE) down

.PHONY: nuke
nuke:  ## Stop the stack and delete volumes
	$(COMPOSE) down -v

# ── quality ────────────────────────────────────────────────────────────────
.PHONY: test
test:  ## Unit + integration tests
	$(UV) run pytest

.PHONY: test-fast
test-fast:  ## Skip anything needing docker
	$(UV) run pytest -m "not integration"

.PHONY: lint
lint:  ## ruff + mypy --strict + import-linter
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy packages ml/src simulator/src
	$(UV) run lint-imports

.PHONY: fmt
fmt:  ## Auto-fix formatting and the mechanical lints
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

# ── data & ML ──────────────────────────────────────────────────────────────
.PHONY: simulate
simulate:  ## Generate the synthetic world into data/worlds/seed-<n>/
	$(UV) run python -m fsa_sim.cli generate --seed $${SEED:-42} --out data/worlds

.PHONY: simulate-org
simulate-org:  ## Generate org structure only (works before spend.py is written)
	$(UV) run python -m fsa_sim.cli generate --seed $${SEED:-42} --out data/worlds --org-only

.PHONY: features
features:  ## Assemble the training matrix and list the feature columns
	$(UV) run python -m fsa_ml.cli features --world data/worlds/seed-$${SEED:-42} \
	  --train-as-of 2026-09-01 --eval-as-of 2027-06-01

.PHONY: train-fraud
train-fraud:  ## Train + evaluate the fraud model (needs `make install-ml`)
	$(UV) run --group ml python -m fsa_ml.cli train-fraud --world data/worlds/seed-$${SEED:-42} \
	  --train-as-of 2026-09-01 --eval-as-of 2027-06-01

.PHONY: seed
seed:  ## Load the generated world into Postgres      (M1)
	@echo "not yet — scripts/seed_db.py lands with the expense-api in M1"

.PHONY: eval
eval:  ## Run the eval suite locally                  (M6)
	@echo "not yet — evals/ lands in M6"


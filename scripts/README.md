# scripts/

Operational one-shots. Each is runnable as `uv run python scripts/<name>.py`.

| Script | Lands in | Purpose |
|---|---|---|
| `seed_db.py` | M1 | Load a generated world into Postgres (expenses, org, budgets) |
| `load_fga_model.py` | M1 | Push the OpenFGA authorisation model and write org tuples |
| `teardown.sh` | M7 | `terraform destroy` plus the manual bits it misses |

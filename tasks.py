"""Cross-platform task runner — the Makefile, for machines without `make`.

    uv run python tasks.py                 list the tasks
    uv run python tasks.py serve
    uv run python tasks.py test lint       several in sequence

`make` ships with Linux and macOS developer tools and not with Windows, so the
Makefile silently excludes a whole class of contributor. Every target below is the
same command the Makefile runs; the two files are kept in step deliberately rather
than one generating the other, because a generated Makefile is one more thing to
debug at the moment you least want to.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

UV = "uv"


def _load_env(path: str = ".env") -> None:
    """Read a gitignored `.env` into the environment before any task runs.

    Deliberately not `python-dotenv`: a task runner that needs its own dependency
    installed before it can tell you how to install dependencies is a bootstrap
    problem. Existing environment variables win, so `GOOGLE_API_KEY=... uv run
    python tasks.py serve` still overrides the file, which is what CI and Key Vault
    injection both rely on.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env()

TASKS: dict[str, tuple[str, list[list[str]]]] = {
    # ── environment ─────────────────────────────────────────────────────────
    "install": ("Sync the workspace (all members editable, dev group)", [[UV, "sync"]]),
    "install-ml": (
        "Sync plus the heavy ML group (lightgbm, mlflow, optuna, shap)",
        [[UV, "sync", "--group", "ml"]],
    ),
    # ── quality ─────────────────────────────────────────────────────────────
    "test": ("Run the whole test suite", [[UV, "run", "pytest"]]),
    "test-fast": ("Skip anything needing docker", [[UV, "run", "pytest", "-m", "not integration"]]),
    "lint": (
        "ruff + mypy --strict + import-linter",
        [
            [UV, "run", "ruff", "check", "."],
            [UV, "run", "ruff", "format", "--check", "."],
            [UV, "run", "mypy", "packages", "services/copilot/src", "ml/src", "simulator/src"],
            [UV, "run", "lint-imports"],
        ],
    ),
    "fmt": (
        "Auto-fix formatting and the mechanical lints",
        [
            [UV, "run", "ruff", "format", "."],
            [UV, "run", "ruff", "check", "--fix", "."],
        ],
    ),
    "acl": (
        "Run the ACL isolation suite (ARCHITECTURE.md §10)",
        [[UV, "run", "pytest", "tests/authz/", "-v"]],
    ),
    # ── the service ─────────────────────────────────────────────────────────
    "serve": (
        "Run the copilot on :8080 — open http://localhost:8080 to watch it",
        [[UV, "run", "uvicorn", "copilot_service.app:app", "--port", "8080"]],
    ),
    "traffic": (
        "Drive persona traffic at a running service (N=60 by default)",
        [
            [
                UV,
                "run",
                "python",
                "scripts/generate_traffic.py",
                "--requests",
                os.environ.get("N", "60"),
            ]
        ],
    ),
    # ── data and measurement ────────────────────────────────────────────────
    "simulate": (
        "Generate the synthetic world into data/worlds/seed-<n>/",
        [
            [
                UV,
                "run",
                "python",
                "-m",
                "fsa_sim.cli",
                "generate",
                "--seed",
                os.environ.get("SEED", "42"),
                "--out",
                "data/worlds",
            ]
        ],
    ),
    "measure": (
        "Produce every headline number: ACL, latency, guardrail catch rates",
        [
            [UV, "run", "python", "scripts/measure_platform.py"],
            [UV, "run", "python", "scripts/measure_guardrails.py"],
        ],
    ),
    "features": (
        "Assemble the training matrix and list the feature columns",
        [
            [
                UV,
                "run",
                "python",
                "-m",
                "fsa_ml.cli",
                "features",
                "--world",
                f"data/worlds/seed-{os.environ.get('SEED', '42')}",
                "--train-as-of",
                "2026-09-01",
                "--eval-as-of",
                "2027-06-01",
            ]
        ],
    ),
    "train-fraud": (
        "Train + evaluate the fraud model (run install-ml first)",
        [
            [
                UV,
                "run",
                "--group",
                "ml",
                "python",
                "-m",
                "fsa_ml.cli",
                "train-fraud",
                "--world",
                f"data/worlds/seed-{os.environ.get('SEED', '42')}",
                "--train-as-of",
                "2026-09-01",
                "--eval-as-of",
                "2027-06-01",
            ]
        ],
    ),
    # ── docker stack ────────────────────────────────────────────────────────
    "up": ("Full local stack (needs docker)", [["docker", "compose", "up", "-d"]]),
    "up-lite": (
        "Postgres + Redpanda + Phoenix only",
        [["docker", "compose", "up", "-d", "postgres", "redpanda", "phoenix"]],
    ),
    "down": ("Stop the stack", [["docker", "compose", "down"]]),
}


def usage() -> int:
    print("Argus tasks — uv run python tasks.py <task> [<task> ...]\n")
    width = max(len(name) for name in TASKS)
    for name, (description, _) in TASKS.items():
        print(f"  {name:<{width}}  {description}")
    print("\nEnvironment: SEED=42  N=60 (traffic request count)")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        return usage()

    for name in argv:
        task = TASKS.get(name)
        if task is None:
            print(f"unknown task: {name}\n", file=sys.stderr)
            return usage() or 1
        _, commands = task
        for command in commands:
            print(f"$ {' '.join(command)}", flush=True)
            result = subprocess.run(command)
            if result.returncode != 0:
                return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

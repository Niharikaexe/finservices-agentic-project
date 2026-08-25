"""Cross-platform task runner — the Makefile, for machines without `make`.

    python tasks.py                  list the tasks
    python tasks.py serve
    python tasks.py test lint        several in sequence

Three assumptions this file refuses to make.

**That you have `make`.** It ships with Linux and macOS developer tools and not with
Windows, so a Makefile silently excludes a whole class of contributor.

**That you have `uv`.** uv is faster and understands the workspace natively, so it is
used when present — but `python bootstrap.py && python tasks.py serve` has to work on
a machine with nothing but Python, which is most machines. See `_runner`.

**That the tool is on PATH.** `uv run pytest` finds a console script; without uv the
equivalent is `python -m pytest`, and a bare `pytest` may not resolve at all. `cmd`
holds that difference in one place so no task has to think about it.

Every task is the same command the Makefile runs. The two are kept in step by hand
rather than generating one from the other, because a generated Makefile is one more
thing to debug at the moment you least want to.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_env(path: str = ".env") -> None:
    """Read a gitignored `.env` into the environment before any task runs.

    Deliberately not `python-dotenv`: a task runner that needs its own dependency
    installed before it can tell you how to install dependencies is a bootstrap
    problem. Existing environment variables win, so `GOOGLE_API_KEY=... python
    tasks.py serve` still overrides the file, which is what CI and Key Vault injection
    both rely on.
    """
    try:
        lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env()


def _venv_python() -> Path:
    bin_dir = "Scripts" if sys.platform == "win32" else "bin"
    return ROOT / ".venv" / bin_dir / ("python.exe" if sys.platform == "win32" else "python")


HAVE_UV = shutil.which("uv") is not None
VENV_PYTHON = _venv_python()

#: Tools with no `python -m` entry point, so without uv they are called as scripts
#: from the venv's bin directory rather than through the interpreter.
_SCRIPT_ONLY = frozenset({"lint-imports", "uvicorn"})


def cmd(*args: str) -> list[str]:
    """Build one command for whichever runner is in play."""
    if HAVE_UV:
        return ["uv", "run", *args]

    python = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
    tool, rest = args[0], list(args[1:])
    if tool == "python":
        return [str(python), *rest]
    if tool in _SCRIPT_ONLY:
        script = python.parent / tool
        return [str(script), *rest]
    return [str(python), "-m", tool, *rest]


def install_commands(group: str | None = None) -> list[list[str]]:
    """Install, by whichever route this machine supports.

    uv syncs the whole workspace from the lockfile in one step. Without it the work is
    `bootstrap.py`, which resolves the same set the long way round and explains why in
    its own docstring — chiefly that pip cannot read `[tool.uv.sources]` and would go
    to PyPI looking for `fsa-common`, a name nobody owns.
    """
    if HAVE_UV:
        return [["uv", "sync"] + (["--group", group] if group else [])]
    if group:
        return [[sys.executable, "bootstrap.py"], cmd("pip", "install", f".[{group}]")]
    return [[sys.executable, "bootstrap.py"]]


SEED = os.environ.get("SEED", "42")
WORLD = f"data/worlds/seed-{SEED}"

TASKS: dict[str, tuple[str, list[list[str]]]] = {
    # ── environment ─────────────────────────────────────────────────────────
    "install": ("Install everything into a local venv", install_commands()),
    "install-ml": (
        "Install plus the heavy ML group (lightgbm, mlflow, optuna, shap)",
        install_commands("ml"),
    ),
    # ── quality ─────────────────────────────────────────────────────────────
    "test": ("Run the whole test suite", [cmd("pytest")]),
    "test-fast": ("Skip anything needing docker", [cmd("pytest", "-m", "not integration")]),
    "lint": (
        "ruff + mypy --strict + import-linter",
        [
            cmd("ruff", "check", "."),
            cmd("ruff", "format", "--check", "."),
            cmd("mypy", "packages", "services/copilot/src", "ml/src", "simulator/src"),
            cmd("lint-imports"),
        ],
    ),
    "fmt": (
        "Auto-fix formatting and the mechanical lints",
        [cmd("ruff", "format", "."), cmd("ruff", "check", "--fix", ".")],
    ),
    "acl": (
        "Run the ACL isolation suite (ARCHITECTURE.md §10)",
        [cmd("pytest", "tests/authz/", "-v")],
    ),
    # ── the service ─────────────────────────────────────────────────────────
    "serve": (
        "Run the copilot on :8080 — open http://localhost:8080 to watch it",
        [cmd("uvicorn", "copilot_service.app:app", "--port", "8080")],
    ),
    "traffic": (
        "Drive persona traffic at a running service (N=60 by default)",
        [cmd("python", "scripts/generate_traffic.py", "--requests", os.environ.get("N", "60"))],
    ),
    # ── reading the interaction log ─────────────────────────────────────────
    "trace": (
        "List recent runs from the interaction log (add --run <id> to drill in)",
        [cmd("python", "scripts/trace.py")],
    ),
    "trace-degraded": (
        "Only the runs that refused, and why",
        [cmd("python", "scripts/trace.py", "--degraded")],
    ),
    "cost": (
        "Spend, tokens and latency per model, from the log",
        [cmd("python", "scripts/trace.py", "--cost")],
    ),
    # ── data and measurement ────────────────────────────────────────────────
    "simulate": (
        "Generate the synthetic world into data/worlds/seed-<n>/",
        [cmd("python", "-m", "fsa_sim.cli", "generate", "--seed", SEED, "--out", "data/worlds")],
    ),
    "measure": (
        "Produce every headline number: ACL, latency, guardrail catch rates",
        [
            cmd("python", "scripts/measure_platform.py"),
            cmd("python", "scripts/measure_guardrails.py"),
        ],
    ),
    "features": (
        "Assemble the training matrix and list the feature columns",
        [
            cmd(
                "python",
                "-m",
                "fsa_ml.cli",
                "features",
                "--world",
                WORLD,
                "--train-as-of",
                "2026-09-01",
                "--eval-as-of",
                "2027-06-01",
            )
        ],
    ),
    "train-fraud": (
        "Train + evaluate the fraud model (run install-ml first)",
        [
            cmd(
                "python",
                "-m",
                "fsa_ml.cli",
                "train-fraud",
                "--world",
                WORLD,
                "--train-as-of",
                "2026-09-01",
                "--eval-as-of",
                "2027-06-01",
            )
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
    print("Argus tasks — python tasks.py <task> [<task> ...]\n")
    width = max(len(name) for name in TASKS)
    for name, (description, _) in TASKS.items():
        print(f"  {name:<{width}}  {description}")

    if HAVE_UV:
        runner = "uv"
    elif VENV_PYTHON.exists():
        runner = f".venv ({VENV_PYTHON.relative_to(ROOT)})"
    else:
        runner = f"{sys.executable} — no .venv yet, run: python bootstrap.py"
    print(f"\nRunner:      {runner}")
    print("Environment: SEED=42  N=60 (traffic request count)")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        return usage()

    for name in argv:
        task = TASKS.get(name)
        if task is None:
            print(f"unknown task: {name}\n", file=sys.stderr)
            usage()
            return 1
        _, commands = task
        for command in commands:
            print(f"$ {' '.join(command)}", flush=True)
            try:
                result = subprocess.run(command, cwd=ROOT)
            except FileNotFoundError:
                print(
                    f"\n  cannot find {command[0]!r}.\n"
                    f"  If the project is not installed yet:  python bootstrap.py\n",
                    file=sys.stderr,
                )
                return 1
            if result.returncode != 0:
                return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

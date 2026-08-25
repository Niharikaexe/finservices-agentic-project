"""Set the project up with nothing but Python. No uv, no make, no pip knowledge.

    python3 bootstrap.py

Creates `.venv`, installs everything into it, and prints what to run next. Safe to
re-run; it is idempotent.

Why this file exists rather than a line of README saying `pip install -e .`:

`pyproject.toml` is a **uv workspace**. Each package declares its siblings as ordinary
dependencies — `fsa-gateway` depends on `fsa-common`, `fsa-telemetry`, `fsa-guardrails`
— and `[tool.uv.sources]` is what tells uv those names mean "the directory next door".
pip does not read that table. Run `pip install -e .` and pip goes to PyPI looking for a
package called `fsa-common`.

That is not merely a failure. `fsa-common`, `fsa-authz` and the rest are unregistered
names on a public index, which is the exact setup for a dependency-confusion attack:
anyone may register them, and pip would install a stranger's code into the venv of a
project that handles authorisation decisions. So this script installs the workspace's
own packages with `--no-deps` and resolves the third-party set separately and
explicitly. The `--no-deps` is a security control, not a performance trick.

The third-party list is not hardcoded here. It is computed by reading every member's
`pyproject.toml` at run time and subtracting the internal names, so adding a dependency
to a package cannot leave this script silently stale.
"""

import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
MIN_PYTHON = (3, 12)

#: Install order matters: a package must be installed after the ones it imports, or
#: the editable install records a dependency it cannot see. This is the layering from
#: ARCHITECTURE.md, bottom up — and it is the same order import-linter enforces.
MEMBERS = [
    "packages/common",
    "packages/telemetry",
    "packages/guardrails",
    "packages/authz",
    "packages/retrieval",
    "packages/gateway",
    "services/copilot",
    "ml",
    "simulator",
    "evals",
]


def fail(message: str) -> "None":
    print(f"\n  {message}\n", file=sys.stderr)
    raise SystemExit(1)


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        want = ".".join(map(str, MIN_PYTHON))
        have = ".".join(map(str, sys.version_info[:3]))
        fail(
            f"Argus needs Python {want} or newer; this is {have}.\n"
            f"  You may already have a newer one installed — try:\n"
            f"      python{want} bootstrap.py\n"
            f"  Otherwise: https://www.python.org/downloads/"
        )


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def read_dependencies() -> "tuple[list[str], set[str]]":
    """Return (third-party requirements, internal package names).

    Computed from the files rather than copied from them, so this cannot drift out of
    step with what the packages actually declare.
    """
    import tomllib

    internal: set[str] = set()
    external: set[str] = set()
    documents = []

    for member in MEMBERS:
        path = ROOT / member / "pyproject.toml"
        if not path.exists():
            fail(f"missing {path.relative_to(ROOT)} — is this the repository root?")
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        documents.append(data)
        internal.add(data["project"]["name"].replace("_", "-").lower())

    for data in documents:
        for requirement in data["project"].get("dependencies", []):
            # Strip any version specifier or extra to get the bare distribution name.
            name = requirement.split(";")[0].strip()
            bare = name.split("[")[0]
            for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
                bare = bare.split(separator)[0]
            if bare.strip().replace("_", "-").lower() not in internal:
                external.add(name)

    with (ROOT / "pyproject.toml").open("rb") as handle:
        groups = tomllib.load(handle).get("dependency-groups", {})
    external.update(groups.get("dev", []))

    return sorted(external), internal


def run(command: "list[str]", description: str) -> None:
    print(f"  {description} …", flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        fail(f"failed: {' '.join(command)}")


def main() -> None:
    check_python()
    print(f"\nArgus bootstrap — Python {'.'.join(map(str, sys.version_info[:3]))}\n")

    if not venv_python().exists():
        print(f"  creating {VENV.relative_to(ROOT)} …", flush=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
    else:
        print(f"  reusing {VENV.relative_to(ROOT)}")

    python = str(venv_python())
    external, internal = read_dependencies()

    run([python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"], "upgrading pip")
    run(
        [python, "-m", "pip", "install", "--quiet", *external],
        f"installing {len(external)} third-party packages",
    )

    # `--no-deps` is the security control described in the module docstring: the
    # workspace's own names are unregistered on PyPI, and resolving them there is how
    # a stranger's package ends up inside an authorisation service.
    for member in MEMBERS:
        run(
            [python, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(ROOT / member)],
            f"installing {member}",
        )

    verify = subprocess.run(
        [
            python,
            "-c",
            "import copilot_service.app, fsa_authz, fsa_gateway, fsa_guardrails;print('ok')",
        ],
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0 or "ok" not in verify.stdout:
        fail("installed, but the packages do not import:\n" + verify.stderr[-1500:])

    if not (ROOT / ".env").exists() and (ROOT / ".env.example").exists():
        (ROOT / ".env").write_text((ROOT / ".env.example").read_text(encoding="utf-8"))
        print("\n  wrote .env from .env.example (gitignored — put your key in it)")

    activate = (
        ".venv\\Scripts\\activate" if sys.platform == "win32" else "source .venv/bin/activate"
    )
    print(
        f"""
  Done. {len(internal)} workspace packages, {len(external)} dependencies.

  Next:
      {activate}
      python tasks.py                  list every task
      python tasks.py serve            then open http://localhost:8080

  Without activating, prefix with the venv's python instead:
      {python} tasks.py serve
"""
    )


if __name__ == "__main__":
    main()

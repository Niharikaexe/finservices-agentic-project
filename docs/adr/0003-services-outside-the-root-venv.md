# 0003. Services are not workspace members yet

- **Status:** accepted
- **Date:** 2026-08-22

## Context

`ARCHITECTURE.md` §5 puts each service at `services/<name>/app/` and §8 lists
`services/*` as workspace members. Both are right at the destination, but combining
them today has a concrete problem: six members each exposing a top-level package named
`app` collide in a single shared venv. Whichever installs last wins, and
`import app.main` becomes a coin flip.

A second, smaller problem: `uv` requires every glob-matched workspace member to have a
`pyproject.toml`, so `members = ["services/*"]` fails outright while the service
directories are empty scaffolding.

## Decision

Keep `services/*` out of `[tool.uv.workspace] members` until a service has real code.
Each service directory carries a README naming its milestone. When a service is built,
it joins the workspace with a **distinct importable package name** — `expense_api`,
not `app` — while keeping the directory layout the architecture specifies.

## Consequences

- The workspace resolves and `uv sync` works from the first commit.
- One deviation from §5's `app/` convention, recorded here so it is a decision and not
  a drift. Service Dockerfiles are unaffected: a container installs exactly one service
  and can name it whatever it likes.
- Revisit at M1, when `expense-api` becomes the first real service.

## Alternatives considered

- **Six members all named `app`.** Rejected: shadowing, non-deterministic imports.
- **A venv per service.** Correct for deployment, wrong for the inner loop — it means
  no single `pytest` run can cover a change that spans a package and a service.
- **`src/` layout with namespace packages under one `argus.services.*` root.** Tidy,
  but it makes each service's Docker build install the whole namespace. Reconsider if
  the service count grows.

# 0002. Use a uv workspace monorepo

- **Status:** accepted
- **Date:** 2026-08-22

## Context

Argus is one product built from four shared packages, six services, an ML platform, a
simulator and an eval harness. Those pieces version together — a change to the `Money`
value object should break the fraud feature pipeline immediately, in CI, not three
weeks later when someone bumps a pin.

Constraints that mattered:
- shared code must be **editable** across all consumers during development, with no
  publish step in the loop;
- service Docker images must install a **narrow** dependency slice — the agent image
  has no business carrying LightGBM, and the `gpu` group must never enter a service
  image at all;
- CI must be able to install only what a given job needs, or the pipeline gets slow
  enough that people stop waiting for it.

## Decision

A `uv` workspace at the repo root. Members are `packages/*`, `ml`, `simulator`,
`evals`. Shared packages are declared with `{ workspace = true }` sources, so one
`uv sync` gives one venv with everything editable.

Dependency **groups** (`core`, `agent`, `authz`, `guards`, `ml`, `gpu`, `obs`, `eval`,
`sim`, `dev`) rather than extras: groups are not part of the published metadata, which
is right for an application, and `uv sync --group ml` installs exactly one slice.

`uv.lock` is committed and CI runs with `UV_FROZEN=1`, so a stale lockfile fails the
build rather than silently resolving something different from what was tested.

## Consequences

- Cross-cutting refactors are one commit and one CI run.
- `mypy --strict` sees the whole graph, so a type error in the shared kernel surfaces
  at its call sites rather than at its definition.
- The repo cannot publish `fsa_common` to PyPI without extra work. Acceptable: nothing
  outside this repo consumes it.
- Everyone gets one venv, which is larger than a per-service venv would be. Traded
  deliberately for a fast inner loop; images stay small because they install groups.

## Alternatives considered

- **Poetry + path dependencies.** Works, materially slower, and its group semantics
  are less clean than uv's.
- **Polyrepo, one repo per service.** Correct at organisational scale, wrong here:
  it would turn every shared-kernel change into a version-bump dance for one engineer.
- **pip + requirements.txt per service.** No lock semantics worth the name, and no
  editable cross-linking without manual `-e` entries.

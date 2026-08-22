# 0001. Record architecture decisions

- **Status:** accepted
- **Date:** 2026-08-22

## Context

This project exists to be explained — in interviews, in a writeup, and to a future
version of me who has forgotten why the retriever filters before it ranks. Decisions
made and never recorded become folklore within about a fortnight, and folklore cannot
be defended under questioning.

The specific failure mode being avoided: writing the rationale *retrospectively*, once
the outcome is known. Retrospective rationale is always cleaner than the real thing and
omits the alternatives that looked plausible at the time — which are precisely the
interesting part.

## Decision

Every non-obvious decision gets an ADR in `docs/adr/NNNN-slug.md`, using
`TEMPLATE.md`: Context / Decision / Consequences / Alternatives considered. Written
when the decision is made, not when the milestone closes. ADRs are immutable once
accepted; a changed mind is a new ADR that supersedes the old one.

A decision is "non-obvious" if a competent engineer could reasonably have chosen
differently. Choosing `ruff` is obvious. Choosing OPA *and* OpenFGA rather than one of
them is not.

## Consequences

- Milestone reviews have something concrete to review beyond the diff.
- Slight friction on every decision, which is the point: it discourages accidental
  architecture.
- The interview narrative in M8 assembles itself from these files rather than being
  reconstructed from memory.

## Alternatives considered

- **Comments in code.** Wrong granularity — a cross-cutting decision has no single
  file to live in, and the trade-off gets lost in a diff.
- **A design doc per milestone.** Too coarse. Decisions arrive one at a time.
- **Nothing.** The default, and the reason most projects cannot answer "why".

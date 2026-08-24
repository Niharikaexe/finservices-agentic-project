# 0006. Department visibility recurses over managers, not over viewers

- **Status:** accepted
- **Date:** 2026-08-24

## Context

`ARCHITECTURE.md` §2 sketches the department relation as
`define viewer: member or manager or manager from parent`. Translating that to a real
org tree needs a decision the sketch does not make: `manager from parent` resolves
exactly one level, but the generated orgs are up to four levels deep, so a director two
levels above a team would not see it.

The obvious repair is to make the rule recursive:

```
define viewer: member or manager or viewer from parent
```

This is wrong, and it looks right. `viewer from parent` inherits the *whole* viewer set
downward — so every ordinary member of a parent department becomes a viewer of every
child department. It was shipped, and the first manual probe caught it: an Engineering
employee retrieved the **Operations** department's expense addendum, because Operations
happened to be generated as a child of Engineering.

Nothing about that failure was visible in a passing test suite. It surfaced because a
demo query printed the citations.

## Decision

Recurse over the manager relation, not the viewer relation:

```
define manager_chain: manager or manager_chain from parent
define viewer: member or manager_chain
```

A department's viewers are its own members plus any manager at or above it in the tree.
Managership propagates *down*; membership does not propagate *anywhere*.

`LocalAuthorizationStore._manager_chain` mirrors this exactly, with a depth bound of 32
— a malformed tuple set can contain a parent cycle, and an unbounded walk would hang the
request rather than deny it. Failing closed means bounding the work as well as the
answer.

## Consequences

- `test_no_cross_department_addendum_leakage` now passes over 120 users × every
  readable document, with zero cross-department grants.
- `test_manager_reads_their_own_subtree_only` asserts the positive form: a manager's
  reach is exactly the subtree rooted at their department, never sideways.
- Deep orgs work: a director three levels up sees everything below without a tuple
  being written per department.
- One more relation to maintain, and the local evaluator and the `.fga` file must be
  changed together. `test_parity.py` is what stops them drifting.

## Alternatives considered

- **`manager from parent`, one level, as written in §2.** Correct but incomplete — it
  silently fails on any org deeper than two levels, which is every org we generate.
- **Materialise transitive manager tuples at write time.** Removes the recursion at
  query time, but every org change now requires recomputing a closure, and a missed
  recomputation is a silent authorisation bug. Recursion at read time is cheaper to get
  right.
- **Keep `viewer from parent` and accept parent-department visibility.** Defensible in
  some products; wrong in this one. Department addenda exist precisely because
  departments have limits others should not see.

## Note

This is the ADR to point at when asked "how do you know your authorisation model is
right?" The answer is not "we reviewed it" — it is "we probed it, it was wrong, and the
probe is now in CI."

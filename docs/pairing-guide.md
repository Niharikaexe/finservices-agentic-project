# How we pair on this

The goal is not a finished repo. It is that you can be handed any file in it, in an
interview, and explain why it is the way it is. That changes how we should split the
work.

## The split

**I build the scaffolding. You build the thinking parts.**

| I do | You do |
|---|---|
| Directory layout, packaging, CI, compose, Makefile | The algorithm inside the function |
| Type signatures, docstrings that state the contract | The implementation that satisfies it |
| Test suites that encode acceptance criteria | Making them pass |
| Reference implementations of one item per family | The remaining items in that family |
| Reviewing your diff and naming what is wrong and why | Deciding the trade-offs, writing the ADR |

If I implement the fraud features, you will be able to read them and not defend them.
So the boundary sits at: **anything where a competent engineer could reasonably choose
differently is yours.**

## The loop

Per unit of work — roughly one function or one node:

1. **I stub it.** Signature, docstring with the contract, and a test file that fails
   (or skips, pending your implementation).
2. **You implement.** Ask me for syntax freely — `rng.lognormal` parameterisation,
   pandas rolling windows, LangGraph edge syntax. Asking for syntax is not cheating;
   asking for the design is.
3. **You run `make test`.** Red is information. Read which assertion failed and what
   it was checking before changing anything.
4. **I review.** Not "looks good" — I name what is wrong, why it matters, and what
   fails in production if it ships. If it is right, I say what I would have got wrong.
5. **We commit.** Conventional Commits, small. `feat(sim): generate baseline spend`.
6. **ADR if a real decision was made.** Written now, while you still remember what you
   did not know.

## What to ask me for

Ask freely for:
- **Syntax and API shape** — "how do I express a left-closed rolling window in pandas",
  "what's the LangGraph syntax for a conditional edge"
- **Explanation of anything in the codebase** — including my own code
- **Why a test asserts what it asserts**
- **The interview version** — "how would I explain this in two minutes"
- **A worked example** in a family where you then write the rest

Push back on me when I:
- Implement something you were going to learn from
- Hand you a design decision pre-made without saying why
- Say "obviously" about something that is not obvious

## Explaining as we go

Every change I make comes with:
- **What** changed, in one line
- **Why this way** — the alternative that lost, and why
- **What breaks** if it is wrong
- **The interview angle** where there is one

If I skip these, ask.

## Milestone rhythm

`CLAUDE.md` says: plan the milestone, get confirmation, then build. Concretely —
at the start of each milestone I write the task breakdown into `docs/milestones.md`
and stop. You confirm or redirect. Then we run the loop above per task.

Nothing is done until `make test` and `make lint` are green. Not "done except lint".

## The learning path, ordered

The build order in `ARCHITECTURE.md` §17 is M0→M8 and it is correct for shipping. For
*learning*, the highest-value-per-hour ordering within it is roughly:

1. **The simulator** (§15) — because everything downstream consumes it, and because
   holding ground truth is the thing a real system never has. ← we are here
2. **Point-in-time features** (§11) — the single most interview-relevant MLOps skill,
   and the one most people get wrong in a way they cannot see.
3. **Permission-aware RAG** (§10) — filter-before-rank. Rare, and instantly credible.
4. **Durable HITL with LangGraph** (§9) — "kill the pod mid-approval" is a demo that
   ends the question.
5. **Delayed ground truth** (§11) — explaining why you cannot chart live accuracy is a
   senior signal on its own.
6. Everything else.

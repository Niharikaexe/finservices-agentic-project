# 0007. Ship heuristic rails first, and measure what they miss

- **Status:** accepted
- **Date:** 2026-08-24

## Context

The guardrail layer (§12) needs a prompt-injection rail over receipt OCR text. The
obvious choice is a classifier — NeMo Guardrails ships one, and a fine-tuned model will
outperform regular expressions on paraphrase.

Two things argued for starting with heuristics anyway: they are auditable line by line
(a compliance reviewer can read a regex; they cannot read a weight matrix), and they
cost nothing at inference. But the argument that actually decided it is that **without a
measured heuristic floor, you cannot tell whether a classifier is worth its latency and
its spend.** Teams routinely deploy a model that adds two points over a baseline nobody
measured.

The simulator plants 824 payloads across six families with known ground truth,
including a `benign_lookalike` family of urgent-sounding but legitimate text as a
false-positive control. So both numbers are measurable rather than estimated.

## Decision

Ship pattern-based rails, measure per family, and hold the classifier decision until
the numbers justify it.

**Per-family reporting is not optional.** The first run reported an aggregate catch rate
of 92.7%, which reads as "nearly at the 95% gate, tune it a little". The per-family
breakdown said something completely different:

| Family | Catch rate |
|---|---|
| instruction_override | 100% |
| authority_spoof | 100% |
| delimiter_escape | 100% |
| tool_abuse | 100% |
| **exfiltration** | **66.5%** |

One pattern capped the gap between the verb and the scope word at 30 characters and
missed *"summarise the approval limits configured for all departments"* by four
characters — a third of that family. The aggregate hid it; the breakdown made it
obvious in one line.

After widening the pattern: **100% catch across 699 adversarial payloads, 0 false
positives across 81,739 clean receipts and 125 benign-lookalikes.**

## Consequences

And then the number that matters. A 12-sample held-out set of phrasings the patterns
were never written against — including an `obfuscation` family (leetspeak, letter
spacing, German) the rail was never designed for:

| | Catch rate |
|---|---|
| Payloads the patterns were tuned on | **100%** |
| Held-out novel phrasings | **8.3%** |

**Regular expressions do not generalise.** The 100% is real and it is also nearly
meaningless as a prediction of production behaviour, because production attackers do
not use the phrasings in our fixture file.

That gap is the whole point of this ADR:

- The heuristic rail ships as a **cheap floor**, not as the defence. It catches the
  copy-pasted attacks that make up most real traffic, at zero inference cost.
- A classifier is now a **funded decision with a number attached**: it has to beat 8.3%
  on held-out phrasings, and we will know if it does.
- The eval gate is set on the **held-out** set, not the fixture set. A gate scored
  against the payloads you tuned on is not a gate.
- Every new payload family found in the wild goes into the held-out set first, never
  into the pattern list first.

## Alternatives considered

- **Start with a classifier.** Better generalisation, but with no floor we would never
  learn what it was worth, and it adds latency to every receipt.
- **Report only the aggregate catch rate.** Would have shipped a 66% blind spot in one
  family and called it 92.7%.
- **Report only the fixture catch rate.** Would have shipped "100% injection catch rate"
  onto a slide, which is the kind of number that gets a system trusted more than it
  deserves.

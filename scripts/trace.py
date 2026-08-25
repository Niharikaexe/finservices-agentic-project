"""Read the interaction log back: what happened, in order, for one run.

    uv run python scripts/trace.py                 the last 20 runs, one line each
    uv run python scripts/trace.py --run run-abc…  every stage of one run
    uv run python scripts/trace.py --degraded      only the runs that refused
    uv run python scripts/trace.py --cost          spend and latency, by model

`data/interactions.jsonl` is append-only and one JSON object per line, which makes it
greppable and trivially loadable into pandas or DuckDB for the eval pipeline. This
script is not a new source of truth — it is the same file, formatted for a human who
is trying to answer "why did it say that?" without writing a comprehension first.

Every record carries `run_id`, so one request's rails, retrieval and model call
reassemble into a timeline. That correlation is the whole reason the log exists: a bad
answer is traceable back through its guardrail verdicts to the exact chunks retrieved
and the exact prompt rendered.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_LOG = Path("data/interactions.jsonl")

DIM, BOLD, OFF = "\033[2m", "\033[1m", "\033[0m"
GREEN, YELLOW, RED, CYAN = "\033[32m", "\033[33m", "\033[31m", "\033[36m"


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"no log at {path} — start the service and ask something first")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line means the process died mid-write. Skip it rather
                # than refusing to show the 900 good records before it.
                continue
    return records


def by_run(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        runs[r.get("run_id", "?")].append(r)
    return runs


def _outcome(stages: list[dict[str, Any]]) -> tuple[str, str]:
    """Infer what happened to a run from the stages present.

    The log has no explicit "result" record on purpose — each stage writes only what it
    knows, so nothing has to be updated after the fact and the file stays append-only.
    """
    llm = [s for s in stages if s["kind"] == "llm"]
    blocked = [s for s in stages if s["kind"] == "guardrail" and s.get("action") == "block"]
    if blocked:
        return RED, f"blocked by {blocked[0]['rail']} rail"
    if not llm:
        return YELLOW, "degraded before the model was called"
    return GREEN, "answered"


def summarise(runs: dict[str, list[dict[str, Any]]], *, limit: int, degraded_only: bool) -> None:
    print(f"{BOLD}{'run_id':<24}{'when':<22}{'stages':<8}{'tokens':<10}{'cost':<11}outcome{OFF}")
    shown = 0
    for run_id, stages in list(runs.items())[-limit * 4 :]:
        colour, outcome = _outcome(stages)
        if degraded_only and colour == GREEN:
            continue
        llm = [s for s in stages if s["kind"] == "llm"]
        tokens = sum(s["prompt_tokens"] + s["completion_tokens"] for s in llm)
        cost = sum(s["cost_usd"] for s in llm)
        when = stages[0].get("at", "")[11:19]
        print(
            f"{run_id:<24}{when:<22}{len(stages):<8}{tokens:<10}"
            f"${cost:<10.6f}{colour}{outcome}{OFF}"
        )
        shown += 1
        if shown >= limit:
            break
    if not shown:
        print(f"{DIM}(nothing matched){OFF}")


def detail(stages: list[dict[str, Any]], *, full: bool) -> None:
    """One run, every stage, in the order it happened."""
    for s in sorted(stages, key=lambda r: r.get("at", "")):
        kind = s["kind"]
        at = s.get("at", "")[11:23]
        print(f"\n{DIM}{at}{OFF}  {BOLD}{kind.upper()}{OFF}")

        if kind == "guardrail":
            action = s.get("action", "?")
            colour = {"allow": GREEN, "strip": YELLOW, "block": RED}.get(action, "")
            print(f"  rail      {s['rail']} ({s['direction']})")
            print(f"  action    {colour}{action}{OFF}")
            if s.get("reasons"):
                print(f"  reasons   {', '.join(map(str, s['reasons']))}")

        elif kind == "retrieval":
            # The authorisation numbers are the interesting part: `allowed_documents`
            # is the size of the permitted set BEFORE ranking, which is what proves
            # the ACL was a query predicate and not a filter applied afterwards.
            print(f"  query     {s.get('query', '')[:100]}")
            print(
                f"  asked by  {s.get('principal_role')} in {s.get('tenant_id')}"
                f"/{s.get('principal_department')}   as_of={s.get('as_of')}"
            )
            print(f"  permitted {s.get('allowed_document_count')} documents (pre-ranking)")
            print(
                f"  returned  {len(s.get('returned_chunk_ids', []))} chunks in "
                f"{s.get('latency_ms', 0):.2f} ms"
            )
            for ref, score in zip(
                s.get("returned_citations", []), s.get("scores", []), strict=False
            ):
                print(f"    {CYAN}{ref:<14}{OFF} score={score}")

        elif kind == "llm":
            print(f"  model     {s['provider']}/{s['model']}  template={s['prompt_template_id']}")
            print(
                f"  tokens    {s['prompt_tokens']} in / {s['completion_tokens']} out"
                f"   ${s['cost_usd']:.6f}   {s['latency_ms']:.0f} ms"
            )
            print(f"  params    {s.get('parameters')}  finish={s.get('finish_reason')}")
            body = s.get("rendered_prompt", "")
            out = s.get("output", "")
            if not full:
                body = body[:400] + (" …" if len(body) > 400 else "")
                out = out[:400] + (" …" if len(out) > 400 else "")
            print(f"\n  {DIM}--- prompt (redacted at write time) ---{OFF}")
            print("\n".join("  " + line for line in body.splitlines()))
            print(f"\n  {DIM}--- output ---{OFF}")
            print("\n".join("  " + line for line in out.splitlines()))

        elif kind == "tool":
            print(f"  tool      {s.get('tool')}  outcome={s.get('outcome')}")


def costs(records: list[dict[str, Any]]) -> None:
    llm = [r for r in records if r["kind"] == "llm"]
    if not llm:
        print("no model calls logged yet")
        return
    per: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in llm:
        per[f"{r['provider']}/{r['model']}"].append(r)

    print(
        f"{BOLD}{'model':<30}{'calls':<8}{'in':<10}{'out':<10}{'cost':<12}"
        f"{'p50 ms':<10}{'p95 ms'}{OFF}"
    )
    for name, rs in per.items():
        # Percentiles over clean calls only. A rate-limited call carries its backoff
        # in `latency_ms`, so mixing them in reports the free tier's RPM limit as
        # model latency and makes the number useless for capacity work.
        clean = [r for r in rs if not r.get("retries")] or rs
        lat = sorted(r["latency_ms"] for r in clean)
        p50 = lat[len(lat) // 2]
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        print(
            f"{name:<30}{len(rs):<8}{sum(r['prompt_tokens'] for r in rs):<10}"
            f"{sum(r['completion_tokens'] for r in rs):<10}"
            f"${sum(r['cost_usd'] for r in rs):<11.5f}{p50:<10.0f}{p95:.0f}"
        )
        retried = [r for r in rs if r.get("retries")]
        if retried:
            print(
                f"{DIM}{'':<30}{len(retried)} of these were rate-limited and retried; "
                f"percentiles above exclude them{OFF}"
            )
    total = sum(r["cost_usd"] for r in llm)
    print(f"\n{DIM}total ${total:.5f} over {len(llm)} calls (${total / len(llm):.5f}/call){OFF}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--run", help="show every stage of one run_id (prefix match works)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--degraded", action="store_true", help="only runs that did not answer")
    ap.add_argument("--cost", action="store_true", help="spend and latency by model")
    ap.add_argument("--full", action="store_true", help="do not truncate prompts and outputs")
    args = ap.parse_args()

    records = load(args.log)
    if args.cost:
        costs(records)
        return

    runs = by_run(records)
    if args.run:
        matches = [rid for rid in runs if rid.startswith(args.run)]
        if not matches:
            raise SystemExit(f"no run starting {args.run!r} in {len(runs)} runs")
        for rid in matches[:3]:
            print(f"\n{BOLD}══ {rid} {'═' * max(0, 60 - len(rid))}{OFF}")
            detail(runs[rid], full=args.full)
        return

    summarise(runs, limit=args.limit, degraded_only=args.degraded)
    print(f"\n{DIM}{len(records)} records over {len(runs)} runs in {args.log}")
    print(f"drill in with:  uv run python scripts/trace.py --run <run_id>{OFF}")


if __name__ == "__main__":
    main()

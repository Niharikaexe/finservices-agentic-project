# Runbook — GuardrailBypassSuspected

**Symptom.** `guardrail:trip_rate:5m` for the injection rail is zero for 30 minutes
while `agent_tool_calls_total` shows normal traffic.

**Why this is an alert at all.** A rail that never trips looks identical to a rail
that is working perfectly. Alerting on the *absence* of activity is the only way to
tell them apart, and it is the alert most teams never write.

## Likely causes, in order

1. **The rail was disabled by config** and the change was not noticed. Check the
   guardrail sidecar's env and the last deploy.
2. **An exception is being swallowed** in the rail wrapper, so every call returns
   `allow`. Check error logs for the rail's module; look for a broad `except`.
3. **Input never reaches the rail** — an upstream change routed receipt text around it
   (a new endpoint, a new ingestion path).
4. **Traffic genuinely has no payloads.** Possible, and the least likely explanation
   at normal volume. Verify with 5.

## Diagnosis

```promql
sum by (rail, action) (rate(guardrail_trips_total[30m]))
sum by (rail) (rate(guardrail_duration_seconds_count[30m]))   # is it being CALLED?
```

If the second query is also zero, the rail is not being invoked — that is cause 3 or 4,
not a detection failure.

## Mitigation

5. **Prove it end to end.** Submit a canary receipt containing a known payload from
   `evals/datasets/injections.jsonl` and confirm the trip counter increments. Keep this
   canary running on a schedule — a synthetic payload every 15 minutes turns this alert
   from "suspected" to "confirmed" and removes the ambiguity above.
6. If the rail is down and cannot be restored quickly, **route affected claims to human
   review** rather than auto-approving. Fail closed.

## Prevention

The canary in step 5, running continuously, is the fix. It makes the trip rate never
legitimately zero, so the alert becomes unambiguous.

# Runbook — CrossTenantDenySpike

**Symptom.** `authz:deny_rate:5m` is more than 3x its one-hour-ago baseline.

## Likely causes, in order

1. **A broken tuple write.** Far more common than an attack. An org sync ran, tuples
   were deleted or not written, and legitimate users are now denied. Check for a recent
   org import or department change.
2. **A deploy changed the authorisation model** and relations no longer resolve.
3. **Genuine probing** — someone enumerating object ids to find what exists.
4. **A client bug** retrying a denied request in a loop.

## Diagnosis

```promql
sum by (subject_role, resource_type) (rate(authz_decisions_total{decision="deny"}[5m]))
sum by (resource_type) (rate(authz_decisions_total[5m]))       # denies vs total
```

If denials are concentrated in **one role**, suspect cause 1 or 2. If spread across
many subjects against **one resource type**, suspect cause 3.

Then in Phoenix, filter traces to denied authorisation spans and look at the distinct
subject count. One subject making many denied requests is a client bug; many subjects
each making a few is an org-sync problem.

## Mitigation

- Cause 1 or 2: **re-run the tuple sync** (`scripts/load_fga_model.py`) and verify with
  the ACL suite against production tuples. Do not widen the model to make denials stop.
- Cause 3: the denials mean the system is working. Rate-limit the source, capture the
  probe set, and add the successful shapes to `evals/datasets/acl_probes.jsonl`.

## Prevention

Run the ACL isolation suite against a snapshot of production tuples in CI, not only
against generated ones. A tuple-sync regression should fail a build, not fire an alert.

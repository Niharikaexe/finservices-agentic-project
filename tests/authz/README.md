# ACL isolation suite

`test_isolation.py` lands in **M1** and implements ARCHITECTURE.md §10 in full:
200 synthetic users across 2 tenants, 100 probes, zero unauthorised chunks.

It is deliberately empty right now rather than stubbed — a placeholder test that
passes vacuously is worse than no test, because CI goes green and the suite looks
covered.

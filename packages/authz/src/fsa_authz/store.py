"""The authorisation store interface.

Two implementations sit behind this Protocol:

  `LocalAuthorizationStore`  — evaluates the model in-process. Runs with no server, so
                               the ACL suite is part of the unit test run rather than
                               something that only works when docker is up.
  `OpenFGAStore`             — the real thing over HTTP.

Both must agree. `tests/authz/test_parity.py` runs the identical probe set through
each and asserts identical verdicts, which is what makes the local one trustworthy as
a stand-in rather than a second, divergent implementation of the rules.

`list_objects` is not a convenience. It is the API that makes filter-before-rank
possible: one round trip returns every document this principal may read, and that list
goes into the SQL predicate. Without it you are back to checking each retrieved chunk
after the fact, which is the thing ARCHITECTURE.md §10 forbids.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AuthorizationStore(Protocol):
    """Object-level authorisation. Every method fails closed."""

    def check(self, *, user: str, relation: str, object: str) -> bool:
        """Does `user` have `relation` on `object`? Deny on any uncertainty."""
        ...

    def list_objects(self, *, user: str, relation: str, type: str) -> list[str]:
        """Every object of `type` on which `user` holds `relation`.

        Returns object ids without the `type:` prefix, ready to drop into a query
        predicate. An empty list means "nothing" and is a legitimate answer; an
        unavailable store must raise, never return empty — silently returning nothing
        would look identical to a correct denial and would hide an outage.
        """
        ...

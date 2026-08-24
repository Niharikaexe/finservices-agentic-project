"""OpenFGA over HTTP.

Same Protocol as `LocalAuthorizationStore`, so swapping between them is a
configuration change and nothing in the retriever moves.

The entire point of this class is what it does when things go wrong: **it raises.**
A timeout, a 500, a connection refused — every one of them becomes `FailClosedError`.
There is no `except: return False` and no `except: return True`, and the difference
matters in both directions:

  returning `True`  grants access during an outage. Obviously fatal.
  returning `False` is *also* wrong, because it is indistinguishable from a correct
                    denial. The system looks healthy while nobody can read anything,
                    and you find out from a support ticket rather than an alert.

Raising a distinct error type means the middleware can map it to a 503, the metric
`authz_decisions_total{decision="fail_closed"}` can alert on it, and the runbook has
something to fire on.
"""

from __future__ import annotations

from typing import Any

import httpx

from fsa_authz.tuples import RelationTuple
from fsa_common import DependencyError, FailClosedError


class OpenFGAStore:
    """Client for an OpenFGA server. Fails closed, always."""

    def __init__(
        self,
        base_url: str,
        store_id: str,
        *,
        model_id: str | None = None,
        timeout_seconds: float = 0.5,
        client: httpx.Client | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._store_id = store_id
        self._model_id = model_id
        # A short timeout on purpose. An authorisation check sits in the critical path
        # of every retrieval; waiting five seconds to be told "yes" is its own outage.
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base}/stores/{self._store_id}/{path}"
        try:
            response = self._client.post(url, json=payload)
        except httpx.RequestError as exc:
            raise FailClosedError("openfga unreachable; denying", path=path) from exc
        if response.status_code >= 500:
            raise FailClosedError("openfga error; denying", path=path, status=response.status_code)
        if response.status_code >= 400:
            # A 4xx is our bug — a malformed tuple key or an unknown relation. Still a
            # denial, but a different class of problem, so a different error.
            raise DependencyError(
                "openfga rejected the request", path=path, status=response.status_code
            )
        result: dict[str, Any] = response.json()
        return result

    def _authorization_model(self) -> dict[str, str]:
        return {"authorization_model_id": self._model_id} if self._model_id else {}

    def check(self, *, user: str, relation: str, object: str) -> bool:
        body = self._post(
            "check",
            {
                "tuple_key": {"user": user, "relation": relation, "object": object},
                **self._authorization_model(),
            },
        )
        return bool(body.get("allowed", False))

    def list_objects(self, *, user: str, relation: str, type: str) -> list[str]:
        """One round trip, not N checks.

        This is the call that makes ACL-as-a-query-predicate possible. Doing it as a
        per-chunk `check` would be N round trips *and* would force post-retrieval
        filtering, which §10 forbids because the ranking would already have been
        computed over documents the caller has no right to influence.
        """
        body = self._post(
            "list-objects",
            {"user": user, "relation": relation, "type": type, **self._authorization_model()},
        )
        objects: list[str] = body.get("objects", [])
        return [o.split(":", 1)[1] for o in objects]

    def write(self, tuples: list[RelationTuple]) -> None:
        """Batch tuple writes. OpenFGA caps a transaction, so chunk it."""
        for start in range(0, len(tuples), 100):
            batch = tuples[start : start + 100]
            self._post(
                "write",
                {
                    "writes": {
                        "tuple_keys": [
                            {"user": t.user, "relation": t.relation, "object": t.object}
                            for t in batch
                        ]
                    },
                    **self._authorization_model(),
                },
            )

"""Error taxonomy.

One base class per *category of caller response*, not per module. The point of a
taxonomy is that middleware can map a category to an HTTP status, a metric label and a
log level without knowing which subsystem raised it.

Categories:
  ArgusError          — base; everything we raise deliberately
   ├── ValidationError    caller sent something malformed          -> 422
   ├── NotFoundError      addressed object does not exist          -> 404
   ├── AuthzError         caller may not do this                   -> 403
   │    └── FailClosedError  trust plane unavailable, so we denied -> 503
   ├── ConflictError      state machine refused the transition     -> 409
   ├── GuardrailError     a rail tripped                           -> 422
   └── DependencyError    something downstream broke               -> 502
"""

from __future__ import annotations

from typing import Any


class ArgusError(Exception):
    """Base for every deliberate error. Carries structured context for logging."""

    http_status: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if not self.context:
            return self.message
        rendered = " ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({rendered})"


class ValidationError(ArgusError):
    http_status = 422
    code = "validation_error"


class NotFoundError(ArgusError):
    http_status = 404
    code = "not_found"


class AuthzError(ArgusError):
    http_status = 403
    code = "forbidden"


class FailClosedError(AuthzError):
    """The trust plane could not answer, so we denied.

    This is a *separate* class from AuthzError because it must never be handled by a
    generic `except AuthzError: log and continue`. A fail-closed denial is an outage
    signal as well as a denial. CLAUDE.md: never fall back to permitting.
    """

    http_status = 503
    code = "fail_closed"


class ConflictError(ArgusError):
    http_status = 409
    code = "conflict"


class GuardrailError(ArgusError):
    http_status = 422
    code = "guardrail_tripped"


class DependencyError(ArgusError):
    http_status = 502
    code = "dependency_error"

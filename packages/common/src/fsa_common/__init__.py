"""fsa_common — the Argus shared kernel.

Everything in here is imported by services, ml/ and simulator/ alike. It therefore
imports *nothing* from those layers (enforced by import-linter, see .importlinter).
"""

from fsa_common.errors import (
    ArgusError,
    AuthzError,
    ConflictError,
    DependencyError,
    FailClosedError,
    GuardrailError,
    NotFoundError,
    ValidationError,
)
from fsa_common.logging import configure_logging, get_logger
from fsa_common.money import CurrencyMismatchError, Money, UnknownCurrencyError
from fsa_common.settings import Settings, get_settings

__all__ = [
    "ArgusError",
    "AuthzError",
    "ConflictError",
    "CurrencyMismatchError",
    "DependencyError",
    "FailClosedError",
    "GuardrailError",
    "Money",
    "NotFoundError",
    "Settings",
    "UnknownCurrencyError",
    "ValidationError",
    "configure_logging",
    "get_logger",
    "get_settings",
]

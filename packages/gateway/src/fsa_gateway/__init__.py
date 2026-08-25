"""fsa_gateway — the vendor-neutral model gateway.

Every model call in Argus goes through `ModelGateway`. That choke point is what makes
provider swapping a config change, spend bounded by design, and full interaction
capture unavoidable rather than best-effort.
"""

from fsa_gateway.gateway import BudgetExceededError, ModelGateway, Route
from fsa_gateway.provider import (
    AzureOpenAIProvider,
    Completion,
    EchoProvider,
    GeminiProvider,
    ModelProvider,
    ModelUnavailableError,
    ProviderQuotaExhaustedError,
)

__all__ = [
    "AzureOpenAIProvider",
    "BudgetExceededError",
    "Completion",
    "EchoProvider",
    "GeminiProvider",
    "ModelGateway",
    "ModelProvider",
    "ModelUnavailableError",
    "ProviderQuotaExhaustedError",
    "Route",
]

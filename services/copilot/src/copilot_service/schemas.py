"""Structured outputs. No free-text parsing anywhere in the decision path.

`min_length=1` on `citations` is a hard architectural constraint, not validation
hygiene: **the copilot may not reach a policy answer it cannot ground.** If the model
returns an uncited answer the request fails and falls back to "I don't have a policy
that covers this", which is the safe direction. Never fail open to a confident guess.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    """What a claim points at. `rule_ref` is the id printed in the policy document,
    so a human can go and read the same line."""

    document_id: str = Field(description="The document id shown in the excerpt header.")
    rule_ref: str = Field(
        description="The rule id in square brackets, exactly as printed, e.g. T&E-4.1."
    )
    quoted_span: str = Field(
        max_length=300,
        description="The words from the excerpt that support the claim, copied verbatim.",
    )


class PolicyAnswer(BaseModel):
    answer: str = Field(max_length=1200, description="The answer, in plain language.")
    citations: list[Citation] = Field(
        min_length=1, description="At least one. Every factual claim must be grounded."
    )
    # These descriptions are not documentation — they are sent to the model. The
    # provider renders them into its `responseSchema`, so the field description is the
    # instruction. Without this one, `gemini-3.5-flash-lite` returned 1500000 for
    # Rs 15,000 (correct) and 5000 for Rs 5,000 (major units, wrong by 100x) in two
    # consecutive calls: it was inferring the unit from the field name and getting it
    # right by luck. The repo's money rule is minor units everywhere, so an answer
    # that silently switches unit is the most expensive kind of wrong.
    applicable_limit_minor: int | None = Field(
        default=None,
        description=(
            "The limit in MINOR units — paise, fils, cents. Multiply the amount "
            "printed in the policy by 100. Rs 5,000 is 500000. Rs 15,000 is 1500000. "
            "Null if the excerpts state no numeric limit."
        ),
    )
    confidence: float = Field(
        ge=0, le=1, description="0 to 1. How well the excerpts actually settle the question."
    )

    @model_validator(mode="after")
    def limit_must_be_cited(self) -> PolicyAnswer:
        if self.applicable_limit_minor is not None and not self.citations:
            raise ValueError("a stated limit must carry a citation")
        return self


class AskRequest(BaseModel):
    user_id: str = Field(description="Who is asking. Drives the whole permitted set.")
    question: str = Field(max_length=1000)
    as_of: str = Field(
        default="2026-03-15",
        description=(
            "The EXPENSE date, not today. A March claim is judged against March's "
            "policy. There is no default to now() on purpose."
        ),
    )
    k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    run_id: str
    answer: str
    citations: list[Citation]
    applicable_limit_minor: int | None
    confidence: float

    # ── the observability payload: what happened, not just what was said ────
    principal_role: str
    tenant_id: str
    permitted_documents: int
    retrieved_chunks: int
    guardrail_actions: dict[str, str]
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    degraded: bool = False
    degraded_reason: str | None = None

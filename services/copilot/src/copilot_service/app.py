"""The copilot service.

    uv run uvicorn copilot_service.app:app --port 8080

Endpoints:
    POST /policy/ask   the pipeline
    GET  /metrics      Prometheus scrape target
    GET  /healthz      liveness, including whether the trust plane answers
    GET  /             a live monitoring page, so observability is demoable with no
                       docker running. Grafana reads the same /metrics when it is up.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from copilot_service.dashboard import DASHBOARD_HTML
from copilot_service.pipeline import PolicyPipeline
from copilot_service.schemas import AskRequest, AskResponse
from fsa_common import configure_logging, get_logger, get_settings
from fsa_gateway import (
    EchoProvider,
    GeminiProvider,
    ModelGateway,
    ModelProvider,
    Route,
)
from fsa_guardrails import scan_pii
from fsa_telemetry import InteractionLog
from fsa_telemetry.metrics import EXPENSES

log = get_logger(__name__)
STATE: dict[str, Any] = {}


def _load_dotenv() -> None:
    """Read a gitignored `.env` into the environment if it has not been read already.

    `tasks.py` already does this before launching anything, so this is for the other
    way in: `uvicorn copilot_service.app:app`, which is what a README snippet, a
    Dockerfile CMD or an IDE run configuration will use. Without it the service starts
    perfectly, reports `provider: echo`, and answers every question from canned text —
    a failure that looks exactly like success until someone checks /healthz.

    Real environment variables always win, so a container with the key injected from
    Key Vault ignores any file that happens to be in the image.
    """
    path = Path.cwd() / ".env"
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _provider() -> ModelProvider:
    """Gemini when a key is present, the deterministic stub otherwise.

    The service must start and be demoable with no key at all — an eval suite or a
    smoke test that needs a credential is one that gets skipped. Which provider is
    live is reported by /healthz and stamped on every response, so nobody has to guess
    what they are looking at.
    """
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        # flash-lite by default: the free tier's rate limit is the binding constraint
        # on a demo, not model quality, and a request that 429s teaches nobody
        # anything. Override with ARGUS_LLM_MODEL for a quality comparison.
        model = os.environ.get("ARGUS_LLM_MODEL", "gemini-3.5-flash-lite")
        log.info("using Gemini", model=model)
        return GeminiProvider(model=model)

    log.warning("no GOOGLE_API_KEY set — using the deterministic echo provider")
    return EchoProvider(
        responses={
            # Enough to exercise the full parse → ground → rail path offline.
            "client entertainment": (
                '{"answer": "Client entertainment is capped at Rs 5,000 per claim '
                'under the global policy, unless your department addendum raises it.", '
                '"citations": [{"document_id": "global", "rule_ref": "T&E-4.1", '
                '"quoted_span": "Client Entertainment: claims must not exceed"}], '
                '"applicable_limit_minor": 500000, "confidence": 0.82}'
            ),
            "meals": (
                '{"answer": "Meals are capped at Rs 2,000 per claim.", '
                '"citations": [{"document_id": "global", "rule_ref": "T&E-4.4", '
                '"quoted_span": "Meals: claims must not exceed"}], '
                '"applicable_limit_minor": 200000, "confidence": 0.79}'
            ),
        }
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    _load_dotenv()
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
    from build_authz_world import build  # type: ignore[import-not-found]

    world = build()
    interaction_log = InteractionLog(
        Path("data/interactions.jsonl"),
        # `log_scrub`, not `output`. The redactor runs once per string field per
        # record, so labelling it as an output-rail evaluation makes
        # `guardrail_trips_total{rail="pii",direction="output"}` a function of log
        # volume rather than of model behaviour — which is the series
        # `GuardrailBypassSuspected` alerts on. Same bug as the one fixed in
        # `ModelGateway._pii_redactor`; it had a second home here.
        redactor=lambda t: scan_pii(t, direction="log_scrub").sanitised,
    )
    provider = _provider()
    gateway = ModelGateway(
        {
            "policy_qa": Route(
                "policy_qa",
                provider,
                # Reasoning models draw thinking tokens from this same budget, and
                # thinking runs *before* the first answer token. At 700 the model
                # could spend the entire allowance reasoning and return an empty
                # body, which the pipeline can only degrade. Size it for both.
                max_tokens=int(os.environ.get("ARGUS_MAX_OUTPUT_TOKENS", "2048")),
                max_cost_usd_per_call=0.02,
            )
        },
        interaction_log=interaction_log,
        daily_budget_usd=float(os.environ.get("ARGUS_DAILY_BUDGET_USD", "2.0")),
    )
    STATE.update(
        world=world,
        gateway=gateway,
        log=interaction_log,
        provider_name=provider.name,
        pipeline=PolicyPipeline(
            authz=world.authz,
            retriever=world.retriever,
            gateway=gateway,
            interaction_log=interaction_log,
        ),
    )
    log.info(
        "copilot ready",
        users=len(world.users),
        documents=len(world.documents),
        chunks=world.store.size,
        provider=provider.name,
    )
    yield
    STATE.clear()


app = FastAPI(title="Argus policy copilot", version="0.1.0", lifespan=lifespan)


@app.post("/policy/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    world = STATE["world"]
    try:
        principal = world.principal(request.user_id)
    except StopIteration:
        raise HTTPException(404, f"unknown user {request.user_id}") from None

    result = STATE["pipeline"].ask(
        request.question,
        principal,
        as_of=date.fromisoformat(request.as_of),
        k=request.k,
    )
    EXPENSES.labels(
        tenant=principal.tenant_id,
        category="policy_qa",
        outcome="degraded" if result.degraded else "answered",
    ).inc()

    return AskResponse(
        run_id=result.run_id,
        answer=result.answer.answer,
        citations=result.answer.citations,
        applicable_limit_minor=result.answer.applicable_limit_minor,
        confidence=result.answer.confidence,
        principal_role=principal.role.value,
        tenant_id=principal.tenant_id,
        permitted_documents=result.retrieval.allowed_document_count,
        retrieved_chunks=len(result.retrieval.chunks),
        guardrail_actions=result.guardrail_actions,
        model=result.model,
        provider=result.provider,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=round(result.cost_usd, 6),
        latency_ms=round(result.latency_ms, 2),
        degraded=result.degraded,
        degraded_reason=result.degraded_reason,
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    world = STATE.get("world")
    if world is None:
        raise HTTPException(503, "not ready")
    # Liveness includes the trust plane. A service that is "up" while authorisation
    # cannot answer is not up in any sense that matters.
    try:
        world.authz.list_objects(user="user:healthcheck", relation="reader", type="policy_document")
        authz_ok = True
    except Exception:
        authz_ok = False
    return {
        "status": "ok" if authz_ok else "degraded",
        "authz": "up" if authz_ok else "down",
        "provider": STATE.get("provider_name"),
        "documents": len(world.documents),
        "chunks": world.store.size,
        "spent_usd": STATE["gateway"].spent_usd,
    }


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/users")
def users() -> list[dict[str, str]]:
    """A small, deliberately varied cast for the demo: one employee per department
    that has an addendum, plus a manager, an auditor and an admin — the four personas
    whose permitted sets differ most."""
    world = STATE["world"]
    picked: list[dict[str, str]] = []
    seen_departments: set[str] = set()
    by_department = {d.department_id: d for d in world.departments}

    for user in world.users:
        if user.role == "employee" and user.department_id not in seen_departments:
            seen_departments.add(user.department_id)
            name = by_department[user.department_id].name
            picked.append(
                {
                    "user_id": user.user_id,
                    "label": f"{user.tenant_id} · {name} · employee",
                }
            )
    for role in ("manager", "auditor", "finance", "admin"):
        match = next((u for u in world.users if u.role == role), None)
        if match:
            picked.append(
                {
                    "user_id": match.user_id,
                    "label": (
                        f"{match.tenant_id} · {by_department[match.department_id].name} · {role}"
                    ),
                }
            )
    return picked[:30]


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    """What the live page polls. The same numbers Grafana would read, without docker."""
    return {
        **STATE["log"].summary(),
        "spent_usd": STATE["gateway"].spent_usd,
        "provider": STATE.get("provider_name"),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML)

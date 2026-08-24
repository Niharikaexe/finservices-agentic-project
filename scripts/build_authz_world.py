"""Assemble a fully-wired world: org, policy corpus, authz tuples, indexed retriever.

One function, used by the ACL suite, the guardrail measurements and the demo, so all
three are looking at exactly the same world. `fsa_sim` is imported here — a script may
depend on the simulator; `packages/*` may not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from fsa_authz import LocalAuthorizationStore, Principal, RelationTuple, Role, build_tuples
from fsa_retrieval import (
    HashingEmbedder,
    InMemoryVectorStore,
    PermissionAwareRetriever,
    ingest,
)
from fsa_sim.world import TENANT_SHAPES, Tenant, WorldConfig
from fsa_sim.world.org import build_org
from fsa_sim.world.policy import PolicyDocument, PolicyRule, build_policy_corpus

WORLD_START = date(2026, 1, 1)
VERSION_SWITCH = date(2026, 7, 1)

_ROLE_MAP = {
    "employee": Role.EMPLOYEE,
    "manager": Role.MANAGER,
    "finance": Role.FINANCE,
    "auditor": Role.AUDITOR,
    "admin": Role.ADMIN,
}


@dataclass
class AuthzWorld:
    tenants: list[Tenant]
    departments: list
    users: list
    documents: list[PolicyDocument]
    rules: list[PolicyRule]
    tuples: list[RelationTuple]
    authz: LocalAuthorizationStore
    retriever: PermissionAwareRetriever
    store: InMemoryVectorStore

    def principal(self, user_id: str) -> Principal:
        user = next(u for u in self.users if u.user_id == user_id)
        return Principal(
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            role=_ROLE_MAP[user.role],
            department_id=user.department_id,
        )

    def users_of(self, tenant_id: str) -> list:
        return [u for u in self.users if u.tenant_id == tenant_id]


def build(config: WorldConfig | None = None, *, n_tenants: int = 3) -> AuthzWorld:
    config = config or WorldConfig(start_date=WORLD_START, seed=42)
    root = np.random.default_rng(config.seed)
    streams = root.spawn(n_tenants)

    tenants, departments, users, documents, rules = [], [], [], [], []
    for i, shape in enumerate(TENANT_SHAPES[:n_tenants]):
        tenant = Tenant(f"t{i:02d}", f"{shape.name.title()} Co {i}", shape.currency, shape.name)
        depts, people = build_org(tenant, shape, streams[i], world_start=WORLD_START)
        docs, rls = build_policy_corpus(
            tenant, depts, world_start=WORLD_START, version_switch=VERSION_SWITCH
        )
        tenants.append(tenant)
        departments.extend(depts)
        users.extend(people)
        documents.extend(docs)
        rules.extend(rls)

    tuples = build_tuples(
        departments=[(d.department_id, d.tenant_id, d.parent_id) for d in departments],
        users=[(u.user_id, u.tenant_id, u.department_id, u.role) for u in users],
        policy_documents=[(d.document_id, d.tenant_id, d.scope_department_id) for d in documents],
    )

    authz = LocalAuthorizationStore(tuples)
    store = InMemoryVectorStore()
    embedder = HashingEmbedder()
    ingest(rules, store=store, embedder=embedder)
    retriever = PermissionAwareRetriever(authz=authz, store=store, embedder=embedder)

    return AuthzWorld(
        tenants, departments, users, documents, rules, tuples, authz, retriever, store
    )


if __name__ == "__main__":
    from fsa_common import configure_logging

    configure_logging()
    w = build()
    print(f"tenants        {len(w.tenants)}")
    print(f"departments    {len(w.departments)}")
    print(f"users          {len(w.users):,}")
    print(f"policy docs    {len(w.documents)}")
    print(f"policy rules   {len(w.rules)}")
    print(f"authz tuples   {len(w.tuples):,}")
    print(f"indexed chunks {w.store.size}")

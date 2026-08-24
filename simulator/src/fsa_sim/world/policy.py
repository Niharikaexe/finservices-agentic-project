"""The policy corpus — deliberately hard to retrieve correctly.

A corpus where every question has one obviously-relevant document proves nothing. This
one is built to break naive RAG in three specific ways, each of which maps to a test:

**Contradiction.** The global T&E policy caps meals at ₹2,000. The Sales addendum
raises it to ₹5,000 for client entertainment. Both are live, both are retrievable by a
salesperson, and the correct answer applies specificity precedence and *cites the
override*. A retriever that returns only the global cap is confidently wrong.

**Temporal versioning.** Caps change mid-year. A claim dated March must be judged
against March's policy, not today's. So retrieval filters on the expense date, and
`effective_to` is not decoration — it is the difference between a correct verdict and
a plausible one.

**Scope.** A department addendum is readable only by that department (and, via
`viewer from parent`, by managers above it). Engineering must never retrieve the Sales
addendum. This is the property the ACL suite hammers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from fsa_sim.world.entities import Category, Department, Tenant


@dataclass(frozen=True, slots=True)
class PolicyDocument:
    """One versioned policy document, scoped to a department or the whole tenant."""

    document_id: str
    tenant_id: str
    title: str
    scope_department_id: str | None  # None => tenant-wide
    effective_from: date
    effective_to: date | None
    body: str
    doc_type: str  # global_te | addendum | tax | authority


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """A single retrievable rule. `rule_ref` is what a citation points at."""

    rule_ref: str
    document_id: str
    tenant_id: str
    scope_department_id: str | None
    category: Category | None
    cap_minor: int | None
    effective_from: date
    effective_to: date | None
    text: str


# Global caps, in minor units. Version 1 runs from the world start; version 2 tightens
# some and loosens others mid-year, which is what makes the temporal filter testable.
_GLOBAL_CAPS_V1: dict[Category, int] = {
    Category.MEALS: 200_000,  # ₹2,000
    Category.CLIENT_ENTERTAINMENT: 500_000,  # ₹5,000
    Category.TRAVEL_AIR: 2_500_000,
    Category.TRAVEL_GROUND: 150_000,
    Category.LODGING: 1_200_000,
    Category.MILEAGE: 300_000,
    Category.SOFTWARE: 800_000,
    Category.OFFICE_SUPPLIES: 100_000,
    Category.TRAINING: 5_000_000,
    Category.TELECOM: 200_000,
}
_GLOBAL_CAPS_V2: dict[Category, int] = {
    **_GLOBAL_CAPS_V1,
    Category.MEALS: 250_000,  # raised
    Category.TRAVEL_GROUND: 120_000,  # tightened
    Category.SOFTWARE: 1_000_000,  # raised
}

# Department addenda that deliberately CONTRADICT the global policy. The department
# name is matched, not the id, so this survives regeneration with a different seed.
_ADDENDA: dict[str, dict[Category, int]] = {
    "Sales": {Category.CLIENT_ENTERTAINMENT: 1_500_000, Category.MEALS: 500_000},
    "Marketing": {Category.CLIENT_ENTERTAINMENT: 900_000},
    "Engineering": {Category.SOFTWARE: 3_000_000, Category.TRAINING: 8_000_000},
    "Operations": {Category.TRAVEL_GROUND: 400_000, Category.MILEAGE: 600_000},
}


def _rupees(minor: int) -> str:
    return f"Rs {minor / 100:,.0f}"


def _global_body(caps: dict[Category, int], version: int) -> str:
    lines = [
        f"GLOBAL TRAVEL & EXPENSE POLICY - version {version}",
        "",
        "4. CATEGORY LIMITS",
        "   The following per-claim limits apply to all employees unless a",
        "   department addendum states otherwise. Where an addendum and this",
        "   policy conflict, the addendum takes precedence for that department.",
        "",
    ]
    for i, (category, cap) in enumerate(sorted(caps.items(), key=lambda kv: kv[0].value), 1):
        label = category.value.replace("_", " ").title()
        lines.append(f"   T&E-4.{i} {label}: claims must not exceed {_rupees(cap)} per claim.")
    lines += [
        "",
        "5. RECEIPTS",
        "   T&E-5.1 An itemised receipt is required for any claim above Rs 500.",
        "   T&E-5.2 Claims must be submitted within 45 days of the transaction date.",
        "",
        "6. APPROVAL",
        "   T&E-6.1 Claims above Rs 30,000 require approval by the claimant's manager.",
        "   T&E-6.2 No employee may approve their own claim under any circumstance.",
        "   T&E-6.3 Splitting a single expense to remain below a limit is prohibited",
        "           and is treated as a policy violation regardless of intent.",
    ]
    return "\n".join(lines)


def _addendum_body(department: str, caps: dict[Category, int]) -> str:
    lines = [
        f"{department.upper()} DEPARTMENT EXPENSE ADDENDUM",
        "",
        "This addendum OVERRIDES the corresponding limits in the Global Travel &",
        f"Expense Policy for members of the {department} department only.",
        "",
    ]
    for i, (category, cap) in enumerate(sorted(caps.items(), key=lambda kv: kv[0].value), 1):
        label = category.value.replace("_", " ").title()
        lines.append(
            f"   ADD-{department[:3].upper()}-{i} {label}: limit raised to "
            f"{_rupees(cap)} per claim, superseding the global limit."
        )
    lines += [
        "",
        f"   ADD-{department[:3].upper()}-99 All other categories follow the global policy.",
    ]
    return "\n".join(lines)


_TAX_BODY = """REGIONAL TAX TREATMENT

TAX-1.1 GST at 18% applies to services procured within India. The GST component
        must be shown separately on the receipt to be reclaimable.
TAX-1.2 VAT at 5% applies to expenses incurred in the United Arab Emirates.
        A valid TRN must appear on the invoice for the VAT to be reclaimable.
TAX-1.3 Expenses incurred outside India and the UAE are reimbursed gross; no
        input credit is claimed.
TAX-2.1 Entertainment expenses are NOT eligible for input tax credit in either
        jurisdiction, regardless of documentation."""

_AUTHORITY_BODY = """APPROVAL AUTHORITY MATRIX

AUTH-1.1 Up to Rs 30,000       - direct manager.
AUTH-1.2 Rs 30,001 to 200,000  - department head.
AUTH-1.3 Above Rs 200,000      - finance controller AND department head.
AUTH-2.1 Authority may not be delegated downward.
AUTH-2.2 An approver may never approve a claim they submitted, at any amount.
AUTH-2.3 Claims flagged for suspected fraud are removed from the normal
         authority matrix and routed to compliance."""


def build_policy_corpus(
    tenant: Tenant,
    departments: list[Department],
    *,
    world_start: date,
    version_switch: date,
) -> tuple[list[PolicyDocument], list[PolicyRule]]:
    """Build one tenant's corpus: two versions of the global policy, department
    addenda, tax rules and the authority matrix."""
    documents: list[PolicyDocument] = []
    rules: list[PolicyRule] = []
    t = tenant.tenant_id

    # ── global policy, two versions ─────────────────────────────────────────
    for version, caps, start, end in (
        (1, _GLOBAL_CAPS_V1, world_start, version_switch),
        (2, _GLOBAL_CAPS_V2, version_switch, None),
    ):
        doc_id = f"{t}-doc-global-v{version}"
        documents.append(
            PolicyDocument(
                document_id=doc_id,
                tenant_id=t,
                title=f"Global Travel & Expense Policy v{version}",
                scope_department_id=None,
                effective_from=start,
                effective_to=end,
                body=_global_body(caps, version),
                doc_type="global_te",
            )
        )
        for i, (category, cap) in enumerate(sorted(caps.items(), key=lambda kv: kv[0].value), 1):
            label = category.value.replace("_", " ").title()
            rules.append(
                PolicyRule(
                    rule_ref=f"T&E-4.{i}",
                    document_id=doc_id,
                    tenant_id=t,
                    scope_department_id=None,
                    category=category,
                    cap_minor=cap,
                    effective_from=start,
                    effective_to=end,
                    text=f"{label}: claims must not exceed {_rupees(cap)} per claim.",
                )
            )
        for ref, text in (
            ("T&E-5.1", "An itemised receipt is required for any claim above Rs 500."),
            ("T&E-5.2", "Claims must be submitted within 45 days of the transaction date."),
            ("T&E-6.1", "Claims above Rs 30,000 require approval by the claimant's manager."),
            ("T&E-6.2", "No employee may approve their own claim under any circumstance."),
            ("T&E-6.3", "Splitting a single expense to remain below a limit is prohibited."),
        ):
            rules.append(
                PolicyRule(
                    rule_ref=ref,
                    document_id=doc_id,
                    tenant_id=t,
                    scope_department_id=None,
                    category=None,
                    cap_minor=None,
                    effective_from=start,
                    effective_to=end,
                    text=text,
                )
            )

    # ── department addenda — the deliberate contradictions ──────────────────
    for department in departments:
        addendum_caps = _ADDENDA.get(department.name)
        if not addendum_caps:
            continue
        doc_id = f"{t}-doc-add-{department.department_id.rsplit('-', 1)[-1]}"
        documents.append(
            PolicyDocument(
                document_id=doc_id,
                tenant_id=t,
                title=f"{department.name} Expense Addendum",
                scope_department_id=department.department_id,
                effective_from=world_start,
                effective_to=None,
                body=_addendum_body(department.name, addendum_caps),
                doc_type="addendum",
            )
        )
        prefix = department.name[:3].upper()
        for i, (category, cap) in enumerate(
            sorted(addendum_caps.items(), key=lambda kv: kv[0].value), 1
        ):
            label = category.value.replace("_", " ").title()
            rules.append(
                PolicyRule(
                    rule_ref=f"ADD-{prefix}-{i}",
                    document_id=doc_id,
                    tenant_id=t,
                    scope_department_id=department.department_id,
                    category=category,
                    cap_minor=cap,
                    effective_from=world_start,
                    effective_to=None,
                    text=(
                        f"{label}: limit raised to {_rupees(cap)} per claim for "
                        f"{department.name}, superseding the global limit."
                    ),
                )
            )

    # ── tenant-wide reference documents ─────────────────────────────────────
    for doc_type, title, body, refs in (
        (
            "tax",
            "Regional Tax Treatment",
            _TAX_BODY,
            [
                ("TAX-1.1", "GST at 18% applies to services procured within India."),
                ("TAX-1.2", "VAT at 5% applies to expenses incurred in the United Arab Emirates."),
                ("TAX-2.1", "Entertainment expenses are not eligible for input tax credit."),
            ],
        ),
        (
            "authority",
            "Approval Authority Matrix",
            _AUTHORITY_BODY,
            [
                ("AUTH-1.1", "Up to Rs 30,000 - direct manager."),
                ("AUTH-1.2", "Rs 30,001 to 200,000 - department head."),
                ("AUTH-1.3", "Above Rs 200,000 - finance controller and department head."),
                ("AUTH-2.2", "An approver may never approve a claim they submitted."),
            ],
        ),
    ):
        doc_id = f"{t}-doc-{doc_type}"
        documents.append(
            PolicyDocument(
                document_id=doc_id,
                tenant_id=t,
                title=title,
                scope_department_id=None,
                effective_from=world_start,
                effective_to=None,
                body=body,
                doc_type=doc_type,
            )
        )
        for ref, text in refs:
            rules.append(
                PolicyRule(
                    rule_ref=ref,
                    document_id=doc_id,
                    tenant_id=t,
                    scope_department_id=None,
                    category=None,
                    cap_minor=None,
                    effective_from=world_start,
                    effective_to=None,
                    text=text,
                )
            )

    return documents, rules


def applicable_cap(
    rules: list[PolicyRule], *, category: Category, department_id: str, as_of: date
) -> PolicyRule | None:
    """Resolve the cap that actually applies — the answer retrieval must reproduce.

    Precedence: a department addendum beats the global policy (specificity), and only
    rules in force on `as_of` are considered (temporality). This function is the oracle
    the eval set is scored against, so it lives with the corpus rather than in the
    retriever — the retriever must not be able to cheat by calling it.
    """
    live = [
        r
        for r in rules
        if r.category is category
        and r.effective_from <= as_of
        and (r.effective_to is None or r.effective_to > as_of)
    ]
    scoped = [r for r in live if r.scope_department_id == department_id]
    if scoped:
        return min(scoped, key=lambda r: r.rule_ref)
    globals_ = [r for r in live if r.scope_department_id is None]
    return min(globals_, key=lambda r: r.rule_ref) if globals_ else None

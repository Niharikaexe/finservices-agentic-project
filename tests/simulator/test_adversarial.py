"""Ground truth has to be trustworthy or every downstream number is fiction.

These tests assert the properties the ML layer relies on: that fraud is a
transformation rather than a separate population, that labels exist for every claim,
that confirmation is genuinely delayed, and that injection is independent of fraud.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from fsa_sim.adversarial import inject_fraud, inject_payloads
from fsa_sim.world.config import TENANT_SHAPES, FraudConfig, WorldConfig
from fsa_sim.world.entities import ExpenseRecord, FraudLabel, Tenant
from fsa_sim.world.org import build_org
from fsa_sim.world.spend import generate_baseline_expenses
from fsa_sim.world.vendors import build_vendors

TENANT = Tenant(tenant_id="t00", name="Acme", currency="INR", shape="startup")
CONFIG = WorldConfig(start_date=date(2026, 1, 1), end_date=date(2026, 6, 30), seed=7)


@pytest.fixture(scope="module")
def baseline() -> tuple[list, list, list[ExpenseRecord]]:
    rng = np.random.default_rng(5)
    _, users = build_org(TENANT, TENANT_SHAPES[0], rng, world_start=CONFIG.start_date)
    vendors = build_vendors(TENANT, rng, world_start=CONFIG.start_date)
    expenses = generate_baseline_expenses(users, vendors, CONFIG, rng, tenant_currency="INR")
    return users, vendors, expenses


@pytest.fixture(scope="module")
def injected(baseline: tuple) -> tuple[list[ExpenseRecord], list[FraudLabel]]:
    users, _, expenses = baseline
    kept, _ghosts, labels = inject_fraud(
        expenses,
        users,
        FraudConfig(),
        np.random.default_rng(31),
        world_start=CONFIG.start_date,
    )
    return kept, labels


class TestFraudInjection:
    def test_every_claim_is_labelled(
        self, injected: tuple[list[ExpenseRecord], list[FraudLabel]]
    ) -> None:
        """No claim may be unlabelled. An unlabelled row forces the training join to
        infer absence, which is the same mistake as treating unconfirmed as clean."""
        expenses, labels = injected
        assert {e.expense_id for e in expenses} == {label.expense_id for label in labels}

    def test_positive_rate_is_realistically_imbalanced(
        self, injected: tuple[list[ExpenseRecord], list[FraudLabel]]
    ) -> None:
        """1-3% positives. This is why §11 insists on PR-AUC over accuracy: a model
        predicting 'clean' every time scores 97-99% accurate and detects nothing."""
        _, labels = injected
        rate = sum(1 for label in labels if label.is_fraud) / len(labels)
        assert 0.01 <= rate <= 0.06

    def test_some_fraud_is_never_caught(
        self, injected: tuple[list[ExpenseRecord], list[FraudLabel]]
    ) -> None:
        """Undetected fraud is recorded clean and lands in training data as a negative.

        This is label noise, it is realistic, and a dataset without it produces a
        model whose offline PR-AUC overstates real performance by the miss rate.
        """
        _, labels = injected
        positives = [label for label in labels if label.is_fraud]
        caught = [label for label in positives if label.observed_is_fraud]
        assert 0 < len(caught) < len(positives)

    def test_the_observed_label_is_never_optimistic_about_honest_claims(
        self, injected: tuple[list[ExpenseRecord], list[FraudLabel]]
    ) -> None:
        """No honest claim may be recorded as fraud. False accusations would be a
        different (and much worse) kind of noise, and we do not model them."""
        _, labels = injected
        assert not any(label.observed_is_fraud for label in labels if not label.is_fraud)

    def test_clean_verdicts_arrive_faster_than_fraud_verdicts(
        self, injected: tuple[list[ExpenseRecord], list[FraudLabel]]
    ) -> None:
        """The asymmetry that drives the whole delayed-ground-truth design:
        reimbursement takes days, an audit takes weeks. So the recent past always looks
        cleaner than it was."""
        expenses, labels = injected
        submitted = {e.expense_id: e.submitted_at for e in expenses}

        def lag(label: FraudLabel) -> float:
            assert label.confirmed_at is not None
            return (label.confirmed_at - submitted[label.expense_id]).days

        clean = [lag(label) for label in labels if not label.observed_is_fraud]
        fraud = [lag(label) for label in labels if label.observed_is_fraud]
        assert np.median(fraud) > np.median(clean) * 2

    def test_every_claim_eventually_resolves(
        self, injected: tuple[list[ExpenseRecord], list[FraudLabel]]
    ) -> None:
        _, labels = injected
        assert all(label.confirmed_at is not None for label in labels)

    def test_confirmation_always_follows_submission(
        self, injected: tuple[list[ExpenseRecord], list[FraudLabel]]
    ) -> None:
        """The defining constraint. If a label could be confirmed before the claim was
        submitted, `label_as_of` would leak it into training."""
        expenses, labels = injected
        submitted = {e.expense_id: e.submitted_at for e in expenses}
        for label in labels:
            if label.confirmed_at is not None:
                assert label.confirmed_at > submitted[label.expense_id]

    def test_negatives_carry_no_typology(
        self, injected: tuple[list[ExpenseRecord], list[FraudLabel]]
    ) -> None:
        _, labels = injected
        assert all(label.typology is None for label in labels if not label.is_fraud)

    def test_fraud_does_not_shift_the_honest_amount_distribution(
        self, baseline: tuple, injected: tuple[list[ExpenseRecord], list[FraudLabel]]
    ) -> None:
        """The ADR-0004 property, asserted.

        The *honest* claims that survive injection must be distributed exactly as they
        were before. If injecting fraud perturbed the honest population, a model could
        learn the perturbation instead of the crime.
        """
        _, _, original = baseline
        expenses, labels = injected
        honest_ids = {label.expense_id for label in labels if not label.is_fraud}
        honest = [e for e in expenses if e.expense_id in honest_ids]
        by_id = {e.expense_id: e for e in original}
        # Every surviving honest claim is byte-identical to the one generated.
        assert all(by_id[e.expense_id] == e for e in honest if e.expense_id in by_id)

    def test_splits_land_under_the_approval_threshold(
        self, injected: tuple[list[ExpenseRecord], list[FraudLabel]]
    ) -> None:
        expenses, labels = injected
        by_id = {e.expense_id: e for e in expenses}
        threshold = FraudConfig().approval_threshold_minor
        splits = [
            by_id[label.expense_id]
            for label in labels
            if label.typology is not None and label.typology.value == "split"
        ]
        assert splits, "no split typology generated — check the typology mix"
        assert all(claim.amount_minor < threshold for claim in splits)

    def test_duplicates_share_the_originals_perceptual_hash(
        self, injected: tuple[list[ExpenseRecord], list[FraudLabel]]
    ) -> None:
        """The signal the duplicate feature family exists to find."""
        expenses, labels = injected
        by_id = {e.expense_id: e for e in expenses}
        dupes = [
            label
            for label in labels
            if label.typology is not None and label.typology.value == "duplicate"
        ]
        assert dupes
        for label in dupes:
            original_id = label.linked_expense_ids[0]
            assert by_id[label.expense_id].receipt_phash == by_id[original_id].receipt_phash
            # ...but the amount was altered, so an exact-match check would miss it.
            assert by_id[label.expense_id].amount_minor != by_id[original_id].amount_minor


class TestInjectionPayloads:
    def test_payloads_land_in_the_receipt_text(self, baseline: tuple) -> None:
        _, _, expenses = baseline
        tainted, labels = inject_payloads(expenses, 0.02, np.random.default_rng(3))
        by_id = {e.expense_id: e for e in tainted}
        assert labels
        for label in labels:
            assert label.payload_text in by_id[label.expense_id].receipt_ocr_text

    def test_only_targeted_receipts_are_modified(self, baseline: tuple) -> None:
        _, _, expenses = baseline
        tainted, labels = inject_payloads(expenses, 0.02, np.random.default_rng(3))
        targeted = {label.expense_id for label in labels}
        original = {e.expense_id: e.receipt_ocr_text for e in expenses}
        for claim in tainted:
            if claim.expense_id not in targeted:
                assert claim.receipt_ocr_text == original[claim.expense_id]

    def test_the_false_positive_control_family_exists(self, baseline: tuple) -> None:
        """`benign_lookalike` payloads are urgent-sounding but legitimate. Without
        them, a rail that trips on the word 'approve' scores 100% catch rate and is
        useless in production."""
        _, _, expenses = baseline
        _, labels = inject_payloads(expenses, 0.05, np.random.default_rng(9))
        assert any(label.payload_family == "benign_lookalike" for label in labels)

    def test_injection_is_independent_of_fraud(
        self, baseline: tuple, injected: tuple[list[ExpenseRecord], list[FraudLabel]]
    ) -> None:
        """If injections concentrated on fraudulent claims, the fraud model could
        detect fraud by looking for the word SYSTEM in the receipt — and the injection
        eval would be measuring the fraud model."""
        expenses, labels = injected
        fraud_ids = {label.expense_id for label in labels if label.is_fraud}
        _, inj_labels = inject_payloads(expenses, 0.05, np.random.default_rng(13))
        injected_ids = {label.expense_id for label in inj_labels}
        overlap = len(fraud_ids & injected_ids) / max(len(injected_ids), 1)
        base_rate = len(fraud_ids) / len(expenses)
        assert overlap < base_rate + 0.05  # no meaningful correlation

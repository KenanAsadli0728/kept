import re
import pandas as pd
from kept import engine
from kept.policy import notes

PHRASES = re.compile(r"sorry|apolog|we promise|do better|regret|inconvenien|unfortunate|make it up|overpay", re.I)


def test_pipeline_writes_entries(world, ledger):
    assert len(ledger) > 200 and {"settled", "review", "open"} <= set(ledger.status)


def test_rerun_is_idempotent_no_double_pay(world, ledger):
    before = len(ledger)
    total = ledger.value_usd.sum()
    again = engine.run_pipeline(world["eng"], models=world["models"])
    now = pd.read_sql("SELECT * FROM ledger_entries", world["eng"])
    assert again["inserted"] == 0 and len(now) == before and abs(now.value_usd.sum() - total) < 1e-6
    assert not now.duplicated(["contract_type", "source_ref"]).any()


def test_maya_story(world):
    L = pd.read_sql("SELECT * FROM ledger_entries WHERE customer_id=1", world["eng"])
    assert {"late_delivery", "price_protection", "expiring_points", "unused_subscription", "devaluation_true_up"} <= set(L.contract_type)
    assert (L.status == "settled").all()
    late = L[L.contract_type == "late_delivery"].iloc[0]
    assert late.settlement_mode == "explicit" and late.value_usd == 3.0          # 47 min, 5% of 62.40 capped at $3
    assert L[L.contract_type == "price_protection"].iloc[0].value_usd == 20.0
    assert L[L.contract_type == "unused_subscription"].iloc[0].settlement_mode == "silent"      # experimental => silent


def test_no_apology_or_promise_language_in_any_note(ledger):
    assert not ledger.note.str.contains(PHRASES).any()


def test_guardrail_validator_rejects_bad_notes():
    for bad in ["We're sorry about the delay. $3.00 credited.", "We promise to do better. $3.00 credited.",
                "Apologies! $3.00 credited.", "We regret the inconvenience: $3.00", "$3.00 credited. You were overpaying."]:
        assert not notes.validate_note(bad, "$3.00")
    assert notes.validate_note("Order #5 arrived 47 min late. $3.00 credited.", "$3.00")
    assert not notes.validate_note("Order #5 arrived late. $9.00 credited.", "$3.00")        # amount must be preserved
    assert not notes.validate_note("x" * 400, None)


def test_every_entry_has_reasons_and_valid_status(ledger):
    assert ledger.reasons.notna().all() and (ledger.reasons.str.len() > 2).all()
    assert set(ledger.status) <= {"open", "settled", "review", "denied"}


def test_engine_never_auto_denies(ledger):
    assert (ledger.status == "denied").sum() == 0


def test_holdout_entries_are_not_auto_settled(ledger):
    h = ledger[ledger.experiment_arm == "holdout"]
    assert len(h) > 0 and not (h.status == "settled").any()
    share = ledger.groupby("customer_id").experiment_arm.first().eq("holdout").mean()
    assert 0.04 < share < 0.2


def test_ring_members_are_routed_to_review(world, ledger):
    gt = pd.read_sql("SELECT ref_id FROM _ground_truth WHERE kind='customer' AND is_ring=1 AND ring_id>=0", world["eng"]).ref_id
    ring = ledger[ledger.customer_id.isin(gt) & ledger.contract_type.isin(["late_delivery", "silent_failure"])]
    assert len(ring) > 10
    assert (ring.status == "review").mean() > 0.9
    flags = pd.read_sql("SELECT customer_id FROM abuse_flags", world["eng"]).customer_id
    assert len(set(gt) & set(flags)) / len(set(gt)) > 0.9


def test_large_payouts_go_to_review(ledger):
    big = ledger[(ledger.value_usd > 50) & (ledger.contract_type != "expiring_points") & (ledger.experiment_arm == "treatment")]
    assert (big.status == "review").all()


def test_incident_discovery_and_silent_victims(world, ledger):
    inc = world["first"]["incidents"]
    assert 2 <= len(inc) <= 3
    assert all(i["n_silent"] > i["n_complained"] for i in inc)
    assert ledger[ledger.contract_type == "incident"].shape[0] > 50


def test_silent_credit_mode_for_unnoticed(ledger):
    pp = ledger[(ledger.contract_type == "price_protection") & (ledger.noticed_flag == 0)]
    assert len(pp) > 0 and (pp.settlement_mode == "silent").all()

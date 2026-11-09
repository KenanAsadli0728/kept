import json
import sys
import pandas as pd
import pytest
from kept import llm
from kept.detector import nlp, incidents
from kept.policy import notes, optimizer

GOOD = json.dumps({"failure_type": "spoiled", "incident_key": {"warehouse": "W3", "carrier": None, "date": "Jan 03"}, "severity": 4, "customer_noticed": True})


# ---------- JSON schema validation ----------
def test_schema_accepts_valid_json_even_inside_prose():
    ex = nlp.parse_llm_json("Sure! Here you go:\n" + GOOD + "\nHope that helps")
    assert ex and ex.failure_type == "spoiled" and ex.incident_key.warehouse == "W3" and ex.severity == 4


@pytest.mark.parametrize("bad", [None, "", "no json here", "{not json}", json.dumps({"failure_type": "spoiled"}),
                                 json.dumps({"failure_type": "banana", "severity": 3, "customer_noticed": True}),
                                 json.dumps({"failure_type": "late", "severity": 9, "customer_noticed": True}),
                                 json.dumps({"failure_type": "late", "severity": 0, "customer_noticed": True}),
                                 json.dumps({"failure_type": "late", "severity": 3, "customer_noticed": "maybe-ish"})])
def test_schema_rejects_invalid(bad):
    assert nlp.parse_llm_json(bad) is None


# ---------- fallback paths ----------
@pytest.fixture
def baseline(world):
    return nlp.load_or_train(world["eng"])


def test_extract_uses_llm_when_valid(monkeypatch, baseline):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: GOOD)
    ex, src = nlp.extract("the milk was warm", baseline)
    assert src == "llm" and ex.failure_type == "spoiled"


@pytest.mark.parametrize("raw", [None, "garbage", '{"failure_type": "x"}'])
def test_extract_falls_back_to_tfidf(monkeypatch, baseline, raw):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: raw)
    ex, src = nlp.extract("my frozen items melted and arrived warm, from warehouse W3", baseline)
    assert src == "fallback" and ex.failure_type == "spoiled" and ex.incident_key.warehouse == "W3" and 1 <= ex.severity <= 5


def test_llm_is_off_by_default_offline():
    assert llm.available() is False and llm.complete("hi") is None


def test_note_llm_guardrail_and_fallback(monkeypatch):
    tmpl = "Order #5 arrived 47 min after the promised time. $3.00 credited."
    monkeypatch.setattr(llm, "complete", lambda *a, **k: None)
    assert notes.make_note(tmpl, "$3.00") == (tmpl, "template")
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "We are so sorry! We promise to do better. $3.00 credited.")
    assert notes.make_note(tmpl, "$3.00") == (tmpl, "template_after_reject")
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "Order #5 was 47 minutes late. $3.00 credited.")
    assert notes.make_note(tmpl, "$3.00")[1] == "llm"
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "Order #5 was late. $4.00 credited.")     # changed amount
    assert notes.make_note(tmpl, "$3.00")[1] == "template_after_reject"


def test_embedding_fallback_when_sentence_transformers_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    E, used = incidents.embed(["melted ice cream", "broken glass box", "never arrived", "late again", "warm milk"], backend="st")
    assert used == "tfidf+svd" and E.shape[0] == 5


def test_expiring_points_fallback_converts_when_no_probability():
    from datetime import datetime, timedelta
    from kept.contracts import expiring_points
    now = datetime(2026, 1, 15, 12)
    d = dict(now=now, points=dict(balance=1800, expires_at=now + timedelta(days=3)), p_not_redeem=None, predicted_reward=None,
             chart_versions=[dict(effective_at=now, rewards={"Iced latte": (150, 4.5)})])
    P = dict(expiry_days=7, min_balance=500, min_p_not_redeem=0.8)
    assert len(expiring_points.evaluate(d, P)) == 1                     # survival model unavailable => convert (offline fallback)
    assert expiring_points.evaluate({**d, "p_not_redeem": 0.3}, P) == []  # likely to redeem on their own => leave alone


# ---------- models ----------
def test_incident_discovery_recovers_planted(world):
    inc = world["first"]["incidents"]
    gt = pd.read_sql("SELECT ref_id, incident_id FROM _ground_truth WHERE kind='order' AND incident_id IS NOT NULL", world["eng"])
    for planted, g in gt.groupby("incident_id"):
        truth = set(g.ref_id)
        best = max(inc, key=lambda i: len(truth & set(i["victim_orders"])))
        assert len(truth & set(best["victim_orders"])) / len(truth) > 0.85


def test_silent_failure_detector_is_good_and_calibrated(world):
    r = world["models"]["silent"].report
    assert r["auc"] > 0.75 and r["reliability"] and r["brier_calibrated"] <= r["brier_raw"] * 1.15
    X = pd.DataFrame([dict(delay_min=5, amount=50, tracking_views=1, returned=1, contact_3d=1, cust_return_rate=0.4,
                           cust_n_orders=5, category=1, carrier=2)])
    why = world["models"]["silent"].reasons(X, k=3)[0]
    assert 1 <= len(why) <= 3


def test_uplift_pilot_and_shrink_in_range(world):
    p = world["models"]["training_report"]["pilot"]
    assert len(p["deciles"]) == 10 and len(p["qini"]["fractions"]) == 21
    assert 0.0 <= world["models"]["uplift_shrink"] <= 1.0


def test_survival_model_quality(world):
    assert world["models"]["survival"].report["auc_within_horizon"] > 0.7


def test_optimizer_respects_budget_allowlist_and_fairness():
    n = 400
    df = pd.DataFrame({"owed_usd": [4.0] * n, "uplift_hat": [0.05 + 0.1 * (i / n) for i in range(n)], "spend_hat": [300.0] * n})
    top = optimizer.allocate(df, budget=20.0, topup_pct=0.5, topup_cap=2.0)
    assert top.sum() <= 20.0 + 1e-6 and top.sum() > 0
    assert top[df.uplift_hat.idxmax()] >= top[df.uplift_hat.idxmin()]                 # high responders first
    assert optimizer.allocate(df, 0.0, 0.5, 2.0).sum() == 0
    groups = pd.DataFrame({"g": ["a", "b"] * (n // 2)}, index=df.index)
    assert optimizer.fairness_audit(df, top, groups, sensitive=["g"])["passed"]
    assert not optimizer.fairness_audit(df.assign(address_hash=1), top, groups)["passed"]    # protected-like input => gate fails

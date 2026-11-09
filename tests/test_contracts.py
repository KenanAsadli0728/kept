from datetime import datetime, timedelta
import yaml
from kept import config
from kept.contracts import late_delivery, price_protection, expiring_points, unused_subscription, devaluation_true_up
from kept.contracts.late_delivery import band_credit

NOW = datetime(2026, 1, 15, 12)
P = {k: v["params"] for k, v in yaml.safe_load(open(config.CONTRACTS_YAML))["contracts"].items()}


def order(delay, amount=100.0, oid=1, age_h=5, items=None, placed_days_ago=2):
    prom = NOW - timedelta(hours=age_h)
    return dict(id=oid, placed_at=NOW - timedelta(days=placed_days_ago), promised_at=prom, delivered_at=prom + timedelta(minutes=delay),
                amount=amount, items=items or [], tracking_views=0)


def data(**kw):
    d = dict(now=NOW, orders=[], price_events={}, subscriptions=[], points=None, redemptions=[], chart_versions=[], contacts=[],
             p_not_redeem=None, predicted_reward=None)
    d.update(kw)
    return d


# ---------------- late delivery: band boundaries ----------------
def test_late_exactly_30_is_not_late():
    assert late_delivery.evaluate(data(orders=[order(30)]), P["late_delivery"]) == []


def test_late_31_is_band0():
    out = late_delivery.evaluate(data(orders=[order(31)]), P["late_delivery"])
    assert len(out) == 1 and out[0].meta["band"] == 0 and out[0].amount == 3.00   # 5% of 100 = 5 -> cap 3


def test_late_60_vs_61_bands_are_graduated_not_cliff():
    a = late_delivery.evaluate(data(orders=[order(60, 50)]), P["late_delivery"])[0]
    b = late_delivery.evaluate(data(orders=[order(61, 50)]), P["late_delivery"])[0]
    assert a.meta["band"] == 0 and b.meta["band"] == 1
    assert b.amount - a.amount <= 1.5          # no big jump across a boundary


def test_late_caps_at_5():
    out = late_delivery.evaluate(data(orders=[order(500, 1000)]), P["late_delivery"])
    assert out[0].amount == 5.00


def test_late_zero_results():
    assert late_delivery.evaluate(data(orders=[order(-10), order(10)]), P["late_delivery"]) == []
    old = order(90); old["delivered_at"] = NOW - timedelta(days=100)
    assert late_delivery.evaluate(data(orders=[old]), P["late_delivery"]) == []
    undelivered = order(90); undelivered["delivered_at"] = None
    assert late_delivery.evaluate(data(orders=[undelivered]), P["late_delivery"]) == []


def test_band_credit_helper():
    assert band_credit(29, 100, P["late_delivery"]["bands"])[2] is None


def test_idempotency_key_stable():
    a = late_delivery.evaluate(data(orders=[order(47)]), P["late_delivery"])[0].source_ref
    b = late_delivery.evaluate(data(orders=[order(47)]), P["late_delivery"])[0].source_ref
    assert a == b == "order:1"


# ---------------- price protection ----------------
def pp_data(drop_days_after, paid=129.0, new=109.0, placed_days_ago=20):
    placed = NOW - timedelta(days=placed_days_ago)
    o = order(0, oid=2, placed_days_ago=placed_days_ago, items=[dict(sku="S1", price_paid=paid)])
    o["placed_at"] = placed
    ev = {"S1": [(placed - timedelta(days=100), paid), (placed + timedelta(days=drop_days_after), new)]}
    return data(orders=[o], price_events=ev)


def test_price_drop_inside_window_refunds_difference():
    out = price_protection.evaluate(pp_data(3, placed_days_ago=10), P["price_protection"])
    assert len(out) == 1 and out[0].amount == 20.00


def test_price_drop_boundary_day_14_in_day_15_out():
    p = P["price_protection"]
    assert len(price_protection.evaluate(pp_data(14, placed_days_ago=30), {**p, "lookback_days": 60})) == 1
    assert price_protection.evaluate(pp_data(15, placed_days_ago=30), {**p, "lookback_days": 60}) == []


def test_price_increase_or_no_event_gives_nothing():
    assert price_protection.evaluate(pp_data(3, new=150.0, placed_days_ago=10), P["price_protection"]) == []
    assert price_protection.evaluate(data(orders=pp_data(3)["orders"]), P["price_protection"]) == []


# ---------------- expiring points ----------------
CHART = [dict(effective_at=NOW - timedelta(days=400), rewards={"Iced latte": (150, 4.5), "Meal box": (600, 18.0)}),
         dict(effective_at=NOW - timedelta(days=20), rewards={"Iced latte": (200, 4.5), "Meal box": (600, 18.0)})]


def pts(balance=1800, days=3):
    return dict(balance=balance, expires_at=NOW + timedelta(days=days))


def test_expiring_points_converts_to_predicted_reward():
    out = expiring_points.evaluate(data(points=pts(), chart_versions=CHART, predicted_reward="Iced latte"), P["expiring_points"])
    assert len(out) == 1 and out[0].meta["reward"] == "Iced latte"
    assert out[0].meta["quantity"] == 2 and out[0].value_usd == 9.0                  # capped at $10 of conversion ...
    assert out[0].meta["extend_points"] == 1400 and out[0].meta["extend_days"] == 90  # ... the rest is extended, not voided


def test_expiring_points_boundaries_and_threshold():
    p = P["expiring_points"]
    assert expiring_points.evaluate(data(points=pts(days=8), chart_versions=CHART), p) == []
    assert expiring_points.evaluate(data(points=pts(days=7), chart_versions=CHART, predicted_reward="Iced latte"), p) != []
    assert expiring_points.evaluate(data(points=pts(balance=499), chart_versions=CHART), p) == []


def test_expiring_points_skips_members_who_will_redeem():
    d = data(points=pts(), chart_versions=CHART, predicted_reward="Iced latte", p_not_redeem=0.2)
    assert expiring_points.evaluate(d, P["expiring_points"]) == []


# ---------------- unused subscription ----------------
def sub(days):
    return dict(id=7, name="Streaming add-on", monthly_price=9.99, last_used_at=NOW - timedelta(days=days))


def test_unused_subscription_boundary_and_credit():
    assert unused_subscription.evaluate(data(subscriptions=[sub(59)]), P["unused_subscription"]) == []
    out = unused_subscription.evaluate(data(subscriptions=[sub(70)]), P["unused_subscription"])
    assert len(out) == 1 and out[0].amount == round(9.99 * 70 / 90 * 0.5 + 1e-9, 2)


def test_unused_subscription_note_has_no_overpayment_alert():
    n = unused_subscription.evaluate(data(subscriptions=[sub(70)]), P["unused_subscription"])[0].note.lower()
    assert "overpay" not in n and "wasting" not in n


# ---------------- devaluation ----------------
def test_devaluation_restores_purchasing_power():
    d = data(points=pts(1800, 100), chart_versions=CHART, redemptions=[dict(reward_name="Iced latte")] * 3)
    out = devaluation_true_up.evaluate(d, P["devaluation_true_up"])
    assert len(out) == 1 and out[0].amount == 600.0           # 1800 * (200/150 - 1) = 600


def test_devaluation_nothing_for_unaffected_reward_or_no_history():
    d = data(points=pts(1800, 100), chart_versions=CHART, redemptions=[dict(reward_name="Meal box")])
    assert devaluation_true_up.evaluate(d, P["devaluation_true_up"]) == []
    assert devaluation_true_up.evaluate(data(points=pts(), chart_versions=CHART), P["devaluation_true_up"]) == []

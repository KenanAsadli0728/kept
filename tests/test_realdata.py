import json
import pandas as pd
import pytest
from kept import config
from kept.realdata import dunnhumby, twcs

DUNN = config.ROOT / "data" / "dunnhumby" / "processed" / "transaction_data.parquet"


def _lines(rows):
    return pd.DataFrame(rows, columns=["household_key", "PRODUCT_ID", "STORE_ID", "DAY", "QUANTITY", "SALES_VALUE"])


def test_price_protection_refunds_a_later_drop_inside_the_window():
    t = _lines([(1, 7, 3, 1, 1, 2.00), (2, 7, 3, 5, 1, 1.50)])           # shopper 1 paid 2.00; day 5 someone paid 1.50
    r = dunnhumby.price_protection(t.assign(WEEK_NO=1, COUPON_DISC=0, RETAIL_DISC=0))
    assert r["lines_with_drop"] == 1 and abs(r["total_refund"] - 0.50) < 1e-9


def test_price_protection_ignores_drops_outside_window_other_stores_and_tiny_changes():
    t = _lines([(1, 7, 3, 1, 1, 2.00), (2, 7, 3, 16, 1, 1.00),          # day 16: outside 14-day window
                (3, 8, 3, 1, 1, 2.00), (4, 8, 9, 3, 1, 1.00),           # other store
                (5, 9, 3, 1, 1, 2.00), (6, 9, 3, 2, 1, 1.95)])         # drop below 5% / $0.25
    r = dunnhumby.price_protection(t.assign(WEEK_NO=1, COUPON_DISC=0, RETAIL_DISC=0))
    assert r["lines_with_drop"] == 0 and r["total_refund"] == 0


def test_price_protection_multiplies_by_quantity_and_uses_lowest_later_price():
    t = _lines([(1, 7, 3, 1, 3, 6.00), (2, 7, 3, 4, 1, 1.80), (3, 7, 3, 6, 1, 1.50)])   # paid 2.00/unit x3; later lows 1.80 then 1.50
    r = dunnhumby.price_protection(t.assign(WEEK_NO=1, COUPON_DISC=0, RETAIL_DISC=0))
    # shopper 1: (2.00 - 1.50) * 3 units = 1.50; shopper 2 also paid more than shopper 3 later: 1.80 - 1.50 = 0.30
    assert r["lines_with_drop"] == 2 and abs(r["total_refund"] - 1.80) < 1e-9


def test_missing_data_is_handled(monkeypatch, tmp_path):
    monkeypatch.setattr(dunnhumby, "DIR", tmp_path)
    monkeypatch.setattr(twcs, "CSV", tmp_path / "nope.csv")
    assert not dunnhumby.available() and not twcs.available()
    with pytest.raises(FileNotFoundError):
        dunnhumby.run(verbose=False)
    with pytest.raises(FileNotFoundError):
        twcs.run(verbose=False)


@pytest.mark.skipif(not DUNN.exists(), reason="Dunnhumby parquet files not in data/dunnhumby/")
def test_real_dunnhumby(monkeypatch, tmp_path):
    monkeypatch.setattr(dunnhumby, "DIR", config.ROOT / "data" / "dunnhumby" / "processed")
    monkeypatch.setattr(dunnhumby, "OUT", tmp_path / "d.json")
    M = dunnhumby.run(verbose=False)
    assert M["overview"]["households"] == 2500
    assert 0.005 < M["price_protection"]["refund_pct_of_sales"] < 0.05
    c = M["campaigns"]
    assert c["campaigns_analysed"] >= 15 and "unmatched" in c and "pooled" in c                # both estimates are always reported
    assert c["pooled"]["ci95_spend"][0] < c["pooled"]["ci95_spend"][1]


def test_twcs_results_schema_if_computed():
    real_out = config.ROOT / "data" / "twcs_metrics.json"
    monkeypatch_out = real_out
    if not real_out.exists():
        pytest.skip("run `python -m kept.realdata.twcs` first")
    M = json.loads(monkeypatch_out.read_text(encoding="utf8"))
    assert M["overview"]["customer_threads"] > 100000 and 0 <= M["effort"]["no_reply_share"] <= 1
    assert M["spikes"]["found"] >= 3 and all(i["top_terms"] for i in M["spikes"]["incidents"])

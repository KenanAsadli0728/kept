import pytest
from kept import config
from kept.olist import analyze

REAL = config.ROOT / "data" / "olist" / "olist_orders_dataset.csv"
DAY = 1440.0


def test_band_credit_is_graduated_and_capped():
    assert analyze.band_credit(0, 100) == 0.0                    # not late
    assert analyze.band_credit(0.5 * DAY, 100) == 5.0            # 5% of 100
    assert analyze.band_credit(0.5 * DAY, 1000) == 9.0           # capped
    assert analyze.band_credit(2 * DAY, 100) == 8.0              # next band, 8%
    assert analyze.band_credit(10 * DAY, 1000) == 15.0           # cap of last band
    assert analyze.band_credit(1 * DAY, 100) == 5.0 and analyze.band_credit(1 * DAY + 1, 100) == 8.0   # boundary


def test_missing_dataset_is_handled(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze, "DIR", tmp_path)
    assert analyze.available() is False
    with pytest.raises(FileNotFoundError):
        analyze.run(verbose=False)


@pytest.mark.skipif(not REAL.exists(), reason="Olist CSVs not downloaded into data/olist/")
def test_real_olist_analysis(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze, "DIR", config.ROOT / "data" / "olist")
    monkeypatch.setattr(analyze, "OUT", tmp_path / "olist_metrics.json")
    M = analyze.run(verbose=False)
    c = M["late_contract"]
    assert 90000 < c["delivered"] < 100000 and 0.06 < c["late_rate"] < 0.10        # the well-known ~8% late rate
    assert 0.3 < c["silent"]["share_no_low_review"] < 0.7
    gap = M["review_effect"]["late_vs_ontime_score_gap"]
    assert gap > 1.0                                                                  # lateness really hurts reviews
    assert M["silent_failure"]["auc"] > 0.58 and M["silent_failure"]["lift_top10pct"] > 1.5
    known = [s for s in M["spikes"] if s["matches_known_event"]]
    assert len(known) >= 2                                                            # Black Friday 2017 + May 2018 recovered
    r = M["repeat_effect"]
    assert r["ci95_pp"][0] < r["ci95_pp"][1] and r["n_customers"] > 50000
    assert M["economics"]["credit_covered_by_retention_ratio"] < 0.5                  # honest finding: retention does not pay for credits

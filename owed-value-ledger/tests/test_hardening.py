import pandas as pd
from ovl import data
from ovl.eval import simulate


def test_missing_order_links_are_inferred(world):
    t = data.load_tables(world["eng"])
    c = t["contacts"]
    assert c.link_inferred.sum() > 100 and c.order_id.notna().all()
    gt = pd.read_sql("SELECT ref_id, true_order_id FROM _ground_truth WHERE kind='contact'", world["eng"]).set_index("ref_id").true_order_id
    inf = c[c.link_inferred].set_index("id")
    assert (inf.order_id == gt.reindex(inf.index)).mean() > 0.75


def test_points_conversion_is_capped_and_remainder_extended(world):
    L = pd.read_sql("SELECT * FROM ledger_entries WHERE contract_type='expiring_points' AND status='settled'", world["eng"])
    assert len(L) > 5 and (L.value_usd <= 10.0 + 1e-9).all()
    adj = pd.read_sql("SELECT * FROM points_adjustments", world["eng"])
    assert len(adj) == len(L) and (adj.extend_days == 90).all()
    maya = L[L.customer_id == 1].iloc[0]
    assert maya.value_usd == 9.0 and "extended 90 days" in maya.note


def test_pipeline_does_not_reconvert_extended_points(world):
    from ovl import engine
    assert engine.run_pipeline(world["eng"], models=world["models"])["inserted"] == 0


def test_metrics_report_power_robustness_and_abuse_by_group(world):
    M = simulate.compute_metrics(world["eng"], models=world["models"])
    assert M["power"]["customers_needed_per_arm"] > M["power"]["holdout_customers_now"]
    assert M["uplift_robustness"]["of"] == 8
    g = M["abuse"]["by_group"]
    assert set(g) == {"coordinated_rings", "lone_heavy_abusers", "stealth_mild_abusers"} and g["coordinated_rings"]["combined_recall"] > 0.8
    assert all("full_net" in b for b in M["by_contract"])

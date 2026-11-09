"""Abuse guard = velocity caps + graph ring score + IsolationForest anomaly score, combined into ONE abuse score.

abuse_score = 1 - (1 - 0.8 * ring_strength) * (1 - 0.45 * anomaly_strength)   (noisy-or)
anomaly alone can never flag (max 0.45) - it only AMPLIFIES weak graph evidence. Hard velocity caps also flag.
Flagged => manual review queue, never auto-denied.
"""
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

W_RING, W_ANOM, FLAG_AT = 0.8, 0.45, 0.5


def components(customers: pd.DataFrame) -> dict:
    """Link customers sharing a device / address / payment instrument. Returns {customer_id: component_id}."""
    G = nx.Graph()
    for r in customers.itertuples(index=False):
        for key in (f"d:{r.device_id}", f"a:{r.address_hash}", f"p:{r.payment_hash}"):
            G.add_edge(f"c:{r.id}", key)
    comp = {}
    for i, nodes in enumerate(nx.connected_components(G)):
        for n in nodes:
            if n.startswith("c:"):
                comp[int(n[2:])] = i
    return comp


def graph_edges(customers: pd.DataFrame, member_ids):
    """Customer-customer edges (shared attribute) for visualisation."""
    sub = customers[customers.id.isin(member_ids)]
    edges = set()
    for col in ("device_id", "address_hash", "payment_hash"):
        for _, g in sub.groupby(col):
            ids = sorted(g.id)
            for a, b in zip(ids, ids[1:]):
                edges.add((int(a), int(b), col))
    return [dict(source=a, target=b, kind=k) for a, b, k in edges]


def score(customers: pd.DataFrame, claims30: dict, payout30: dict, params: dict, now) -> pd.DataFrame:
    df = customers[["id", "created_at"]].set_index("id").copy()
    df["claims30"] = pd.Series(claims30).reindex(df.index).fillna(0)
    df["payout30"] = pd.Series(payout30).reindex(df.index).fillna(0.0)
    df["age_days"] = (now - pd.to_datetime(df.created_at)).dt.days
    # --- graph ring strength ---
    comp = components(customers)
    df["comp"] = pd.Series(comp)
    gmean = max(df.claims30.mean(), 1e-6)
    stats = df.groupby("comp").agg(size=("claims30", "size"), mean_claims=("claims30", "mean"))
    stats["ratio"] = stats.mean_claims / gmean
    strength = ((stats.ratio - 1) / (2 * params["ring_score_threshold"])).clip(0, 1)
    strength[stats["size"] < params["min_component_size"]] = 0.0
    df["ring"] = df.comp.map(strength).fillna(0.0)
    df["comp_size"] = df.comp.map(stats["size"])
    # --- anomaly ---
    X = df[["claims30", "payout30", "age_days"]].to_numpy(float)
    iso = IsolationForest(n_estimators=100, contamination=params["anomaly_contamination"], random_state=0).fit(X)
    raw = -iso.score_samples(X)
    df["anomaly"] = pd.Series(raw, index=df.index).rank(pct=True)
    df["anomaly_flag"] = iso.predict(X) == -1
    df["anomaly_strength"] = ((df.anomaly - 0.9) / 0.1).clip(0, 1)
    df["abuse_score"] = 1 - (1 - W_RING * df.ring) * (1 - W_ANOM * df.anomaly_strength)
    df["velocity"] = (df.claims30 > params["max_claims_30d"]) | (df.payout30 > params["max_payout_30d"])
    df["flagged"] = (df.abuse_score >= FLAG_AT) | df.velocity
    df["reason"] = np.where(df.ring >= 0.5, "ring", np.where(df.velocity, "velocity", np.where(df.abuse_score >= FLAG_AT, "anomaly", "")))
    return df

"""Load tables into the per-customer `customer_data` dicts the pure contract functions consume.
NOTE: never reads `_ground_truth` (guarded by tests/test_leakage.py)."""
from collections import defaultdict
import pandas as pd
from sqlalchemy import text

DT = ["placed_at", "promised_at", "delivered_at"]


def _ts(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v) else pd.Timestamp(v).to_pydatetime()


def read(eng, sql, **kw):
    return pd.read_sql(text(sql), eng, **kw)


def infer_order_links(contacts, orders, max_days=7):
    """Tickets without an order reference are linked to the customer's most recent order that had arrived (or been placed)
    before the ticket, within `max_days`. Adds `link_inferred`."""
    c = contacts.copy()
    c["link_inferred"] = c.order_id.isna()
    miss = c[c.order_id.isna()].sort_values("created_at")
    if miss.empty:
        return c
    o = orders[["id", "customer_id", "placed_at", "delivered_at"]].copy()
    o["ref_t"] = o.delivered_at.fillna(o.placed_at)
    o = o.sort_values("ref_t")
    m = pd.merge_asof(miss[["id", "customer_id", "created_at"]], o[["id", "customer_id", "ref_t"]].rename(columns={"id": "inferred"}),
                      left_on="created_at", right_on="ref_t", by="customer_id", direction="backward", tolerance=pd.Timedelta(days=max_days))
    c.loc[c.id.isin(m.id), "order_id"] = c.loc[c.id.isin(m.id), "id"].map(dict(zip(m.id, m.inferred)))
    return c


def load_tables(eng, adjusted=True):
    t = {}
    t["customers"] = read(eng, "SELECT * FROM customers")
    t["orders"] = read(eng, "SELECT * FROM orders", parse_dates=DT)
    t["items"] = read(eng, "SELECT * FROM order_items")
    t["price_events"] = read(eng, "SELECT sku,price,effective_at FROM price_events", parse_dates=["effective_at"])
    t["subs"] = read(eng, "SELECT * FROM subscriptions", parse_dates=["started_at", "last_used_at"])
    t["points"] = read(eng, "SELECT * FROM points_accounts", parse_dates=["expires_at"])
    t["redemptions"] = read(eng, "SELECT * FROM redemptions", parse_dates=["redeemed_at"])
    t["chart"] = read(eng, "SELECT * FROM reward_chart_versions", parse_dates=["effective_at"])
    if adjusted:       # points already converted/extended by the ledger
        adj = read(eng, "SELECT customer_id, SUM(used_points) used, MAX(extend_days) ext FROM points_adjustments GROUP BY customer_id").set_index("customer_id")
        if len(adj):
            pts = t["points"].set_index("customer_id")
            ix = adj.index.intersection(pts.index)
            pts.loc[ix, "balance"] = pts.loc[ix, "balance"] - adj.loc[ix, "used"]
            pts.loc[ix, "expires_at"] = pts.loc[ix, "expires_at"] + pd.to_timedelta(adj.loc[ix, "ext"], unit="D")
            t["points"] = pts.reset_index()
    t["contacts"] = infer_order_links(read(eng, "SELECT * FROM contacts", parse_dates=["created_at"]), t["orders"])
    return t


def chart_versions(chart_df):
    out = []
    for eff, g in chart_df.groupby("effective_at"):
        out.append(dict(effective_at=eff.to_pydatetime(), rewards={r.reward_name: (int(r.points_cost), float(r.cash_value)) for r in g.itertuples()}))
    return sorted(out, key=lambda v: v["effective_at"])


def iter_customer_data(t, now, only=None):
    """Yield one customer_data dict per customer (optionally just ids in `only`)."""
    items_by_order = defaultdict(list)
    for r in t["items"].itertuples(index=False):
        items_by_order[r.order_id].append(dict(sku=r.sku, price_paid=float(r.price_paid)))
    price_ev = defaultdict(list)
    for r in t["price_events"].itertuples(index=False):
        price_ev[r.sku].append((r.effective_at.to_pydatetime(), float(r.price)))
    orders_by = defaultdict(list)
    for r in t["orders"].to_dict("records"):
        if only is not None and r["customer_id"] not in only:
            continue
        for k in DT: r[k] = _ts(r[k])
        r["items"] = items_by_order.get(r["id"], [])
        orders_by[r["customer_id"]].append(r)
    subs_by, red_by, con_by = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in t["subs"].to_dict("records"):
        r["last_used_at"] = _ts(r["last_used_at"]); r["started_at"] = _ts(r["started_at"]); subs_by[r["customer_id"]].append(r)
    for r in t["redemptions"].to_dict("records"): red_by[r["customer_id"]].append(r)
    for r in t["contacts"].to_dict("records"):
        r["created_at"] = _ts(r["created_at"]); con_by[r["customer_id"]].append(r)
    pts = {r["customer_id"]: dict(balance=int(r["balance"]), expires_at=_ts(r["expires_at"])) for r in t["points"].to_dict("records")}
    cv = chart_versions(t["chart"])
    for c in t["customers"].to_dict("records"):
        if only is not None and c["id"] not in only:
            continue
        cid = c["id"]
        yield dict(customer=c, now=now, orders=orders_by.get(cid, []), price_events=price_ev, subscriptions=subs_by.get(cid, []),
                   points=pts.get(cid), redemptions=red_by.get(cid, []), chart_versions=cv, contacts=con_by.get(cid, []),
                   p_not_redeem=None, predicted_reward=None)

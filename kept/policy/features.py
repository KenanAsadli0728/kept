"""Customer-level features for the uplift model / optimizer. Operational tables only (never `_ground_truth`)."""
import pandas as pd

SEG = {"new": 0, "regular": 1, "vip": 2}


def customer_features(t, now, exposure: pd.DataFrame = None) -> pd.DataFrame:
    """exposure: index customer_id, columns exposure_usd, share_noticed (from IOUs)."""
    c = t["customers"].set_index("id")
    o = t["orders"]
    o12 = o[o.placed_at >= now - pd.Timedelta(days=365)]
    g = o12.groupby("customer_id").agg(n_orders=("id", "size"), avg_amount=("amount", "mean"))
    late = o12[(o12.delivered_at - o12.promised_at).dt.total_seconds() > 1800].groupby("customer_id").size().rename("n_late")
    con = t["contacts"].groupby("customer_id").size().rename("n_contacts")
    df = pd.DataFrame(index=c.index).join(g).join(late).join(con)
    df["seg"] = c.segment.map(SEG)
    df["tenure_days"] = (now - pd.to_datetime(c.created_at)).dt.days
    df = df.fillna({"n_orders": 0, "avg_amount": 40.0, "n_late": 0, "n_contacts": 0})
    df["spend_hat"] = df.n_orders * df.avg_amount           # 12-month spend estimate
    if exposure is not None:
        df = df.join(exposure)
    for col in ("exposure_usd", "share_noticed"):
        if col not in df: df[col] = 0.0
    df[["exposure_usd", "share_noticed"]] = df[["exposure_usd", "share_noticed"]].fillna(0.0)
    return df

from datetime import timedelta
from .base import IOU, money


def evaluate(data, params):
    """Auto-convert expiring points into the member's likely reward - but only if the survival model
    says they probably will NOT redeem on their own (p_not_redeem; None => assume yes, offline fallback)."""
    p = data.get("points")
    if not p:
        return []
    left = p["expires_at"] - data["now"]
    if not (timedelta(0) <= left <= timedelta(days=params["expiry_days"])) or p["balance"] < params["min_balance"]:
        return []
    pnr = data.get("p_not_redeem")
    if pnr is not None and pnr < params["min_p_not_redeem"]:
        return []
    chart = data["chart_versions"][-1]["rewards"] if data["chart_versions"] else {}
    name = data.get("predicted_reward")
    if name not in chart or chart[name][0] > p["balance"]:
        afford = [(c, n) for n, (c, cash) in chart.items() if c <= p["balance"]]
        if not afford:
            return []
        name = max(afford)[1]
    cap = params.get("max_convert_usd", 1e9)
    if chart[name][1] > cap:                       # predicted reward is worth more than the cap: fall back to the best affordable one under it
        fit = [(cash_, n) for n, (c_, cash_) in chart.items() if cash_ <= cap and c_ <= p["balance"]]
        if not fit:
            return []
        name = max(fit)[1]
    cost, cash = chart[name]
    k = int(max(1, min(p["balance"] // cost, params.get("max_convert_usd", 1e9) // cash)))   # convert a capped amount ...
    used = k * cost
    remaining = int(p["balance"] - used)                                                    # ... and extend the rest instead of voiding it
    ext = params.get("extend_days", 0)
    tail = f" Remaining {remaining:,} points extended {ext} days." if remaining > 0 and ext else ""
    return [IOU("expiring_points", f"points:{p['expires_at']:%Y%m%d}", float(used), "POINTS", money(k * cash), data["now"],
                f"{used:,} of your points were due to expire on {p['expires_at']:%b %d}. Converted into {k} x {name}.{tail}",
                dict(reward=name, quantity=k, points_used=int(used), extend_points=remaining if ext else 0, extend_days=ext,
                     days_left=round(left.total_seconds() / 86400, 1), p_not_redeem=pnr))]

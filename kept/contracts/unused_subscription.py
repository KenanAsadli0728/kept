from .base import IOU, money


def evaluate(data, params):
    """EXPERIMENTAL. Pause + prorated credit, framed as value-preserving. Never emits an 'overpaying' alert."""
    out = []
    for s in data["subscriptions"]:
        days = (data["now"] - s["last_used_at"]).days
        if days >= params["unused_days"]:
            credit = money(s["monthly_price"] * min(1.0, days / params["proration_days"]) * params.get("credit_pct", 1.0))
            out.append(IOU("unused_subscription", f"sub:{s['id']}:{s['last_used_at']:%Y%m%d}", credit, "USD", credit, data["now"],
                           f"{s['name']} paused after {days} days without use. ${credit:.2f} prorated credit applied. Resume any time.",
                           dict(sub_id=s["id"], days_unused=days)))
    return out

from datetime import timedelta
from .base import IOU, money


def band_credit(delay_min, amount, bands):
    """Graduated bands (lo, hi]; returns (credit, uncapped, band_index) or (0, 0, None)."""
    for i, b in enumerate(bands):
        if b["lo"] < delay_min <= b["hi"]:
            unc = amount * b["pct"]
            return min(b["cap"], unc), unc, i
    return 0.0, 0.0, None


def evaluate(data, params):
    out = []
    horizon = data["now"] - timedelta(days=params["lookback_days"])
    for o in data["orders"]:
        if o["delivered_at"] is None or o["delivered_at"] < horizon:
            continue
        delay = (o["delivered_at"] - o["promised_at"]).total_seconds() / 60
        credit, unc, band = band_credit(delay, o["amount"], params["bands"])
        credit = money(credit)
        if band is None or credit < params["min_credit"]:
            continue
        out.append(IOU("late_delivery", f"order:{o['id']}", credit, "USD", credit, o["delivered_at"],
                       f"Order #{o['id']} arrived {round(delay)} min after the promised time. ${credit:.2f} credited.",
                       dict(order_id=o["id"], delay_min=round(delay, 1), band=band, uncapped=money(unc),
                            tracking_views=o.get("tracking_views", 0))))
    return out

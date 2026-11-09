from datetime import timedelta
from .base import IOU, money


def evaluate(data, params):
    out = []
    horizon = data["now"] - timedelta(days=params["lookback_days"])
    win = timedelta(days=params["window_days"])
    for o in data["orders"]:
        if o["placed_at"] < horizon:
            continue
        end = min(o["placed_at"] + win, data["now"])
        for it in o["items"]:
            ev = [(t, p) for (t, p) in data["price_events"].get(it["sku"], []) if o["placed_at"] < t <= end]
            if not ev:
                continue
            low = min(p for _, p in ev)
            diff = money(it["price_paid"] - low)
            if diff >= params["min_refund"]:
                t_drop = min(t for t, p in ev if p == low)
                out.append(IOU("price_protection", f"order:{o['id']}:{it['sku']}", diff, "USD", diff, t_drop,
                               f"Price of {it['sku']} fell within {params['window_days']} days of purchase. ${diff:.2f} refunded.",
                               dict(order_id=o["id"], sku=it["sku"])))
    return out

from datetime import timedelta
from collections import Counter
from .base import IOU, money


def evaluate(data, params):
    """When a reward chart makes the member's usual reward more expensive, restore the lost purchasing power."""
    vs, p = data["chart_versions"], data.get("points")
    if len(vs) < 2 or not p or not data["redemptions"]:
        return []
    fav = Counter(r["reward_name"] for r in data["redemptions"]).most_common(1)[0][0]
    out = []
    for old, new in zip(vs[:-1], vs[1:]):
        if new["effective_at"] < data["now"] - timedelta(days=params["lookback_days"]) or new["effective_at"] > data["now"]:
            continue
        if fav not in old["rewards"] or fav not in new["rewards"]:
            continue
        c0, _ = old["rewards"][fav]
        c1, cash = new["rewards"][fav]
        if c1 <= c0:
            continue
        restore = int(round(p["balance"] * min(params["max_restore_pct_of_balance"], c1 / c0 - 1)))
        if restore <= 0:
            continue
        out.append(IOU("devaluation_true_up", f"chart:{new['effective_at']:%Y%m%d}", float(restore), "POINTS",
                       money(restore * cash / c1), new["effective_at"],
                       f"{fav} now costs {c1} points (was {c0}). {restore:,} points restored to your balance.",
                       dict(reward=fav, old_cost=c0, new_cost=c1)))
    return out

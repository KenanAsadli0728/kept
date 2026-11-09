"""Settlement policy: noticed signal -> explicit vs silent, holdout assignment, explanations, settlement timing."""
import zlib
from datetime import datetime, timedelta

from .. import config

CASH_TYPES = {"late_delivery", "price_protection", "silent_failure", "incident", "unused_subscription"}
CLAIM_TYPES = {"late_delivery", "silent_failure", "incident"}   # velocity-capped "claim-like" events


def next_nightly(t: datetime, hour=2) -> datetime:
    """The nightly batch (02:00) that first sees an event."""
    n = t.replace(hour=hour, minute=0, second=0, microsecond=0)
    return n if n > t else n + timedelta(days=1)


def contacted(iou, data):
    """Customer contacted support about THIS order (contacts are linked to orders)."""
    oid = iou.meta.get("order_id")
    return oid is not None and any(c.get("order_id") == oid for c in data["contacts"])


def noticed_signal(iou, data, params) -> bool:
    """Did the customer very likely notice the failure?"""
    t = iou.contract_type
    if t == "incident":
        return bool(iou.meta.get("complained"))
    if t == "devaluation_true_up":
        return True                                     # public, high-visibility change
    if contacted(iou, data):
        return True
    if t == "late_delivery":
        return iou.meta.get("delay_min", 0) > params["notice_delay_min"] or iou.meta.get("tracking_views", 0) >= params["notice_tracking_views"]
    if t == "silent_failure":
        return False                                    # no contact: undetected by the customer's own signals
    return False


def decide_mode(noticed: bool, experimental: bool) -> str:
    """Rule 2 (salience): explicit only if noticed; experimental contracts are always silent."""
    return "explicit" if (noticed and not experimental) else "silent"


def assign_arm(customer_id: int, holdout_pct: float) -> str:
    if customer_id == config.MAYA_ID:
        return "treatment"                              # demo persona is never in the holdout
    return "holdout" if (zlib.crc32(f"holdout:{customer_id}".encode()) % 10000) < holdout_pct * 10000 else "treatment"


def reasons_for(iou, data, noticed, detector_reasons=None):
    """Top-3 human-readable reasons stored on each ledger entry."""
    m, t = iou.meta, iou.contract_type
    r = []
    if t == "late_delivery":
        r.append(f"Late {m['delay_min']:.0f} min")
        if m.get("tracking_views"): r.append(f"{m['tracking_views']} prior tracking checks")
        if contacted(iou, data): r.append("support contact about this order")
    elif t == "price_protection":
        r.append(f"Price dropped ${iou.amount:.2f} after purchase"); r.append("inside the protection window")
    elif t == "expiring_points":
        r.append(f"Expires in {m['days_left']:.0f} days")
        if m.get("p_not_redeem") is not None: r.append(f"{m['p_not_redeem']:.0%} modelled chance of not redeeming")
        r.append(f"usually redeems {m['reward']}")
    elif t == "unused_subscription":
        r.append(f"No use for {m['days_unused']} days"); r.append("experimental contract: silent pause")
    elif t == "devaluation_true_up":
        r.append(f"{m['reward']} cost rose {m['old_cost']} -> {m['new_cost']} points"); r.append("reward chart change")
    elif t == "silent_failure":
        r = list(detector_reasons or [])[:3] or ["model-detected likely failure"]
    elif t == "incident":
        r.append(f"Part of {m['incident_id']}")
        r.append("complained" if m.get("complained") else "did not complain (silent victim)")
    r.append("customer noticed: explicit note" if noticed else "customer unlikely to have noticed: silent credit")
    return r[:3]

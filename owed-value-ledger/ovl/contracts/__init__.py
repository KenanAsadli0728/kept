from . import late_delivery, price_protection, expiring_points, unused_subscription, devaluation_true_up

REGISTRY = {
    "late_delivery": late_delivery.evaluate,
    "price_protection": price_protection.evaluate,
    "expiring_points": expiring_points.evaluate,
    "unused_subscription": unused_subscription.evaluate,
    "devaluation_true_up": devaluation_true_up.evaluate,
}


def evaluate_all(data, contract_rows):
    out = []
    for cid, fn in REGISTRY.items():
        row = contract_rows.get(cid)
        if row and row["active"]:
            for iou in fn(data, row["params"]):
                iou.customer_id = data["customer"]["id"]
                iou.source_ref = f"c{iou.customer_id}/{iou.source_ref}"     # unique per customer -> idempotency key
                iou.meta["experimental"] = row["experimental"]
                out.append(iou)
    return out

"""The HIDDEN response function used by the simulation (ground truth, evaluation only).

Retention(90d) = base_i + tau_i * g(credit) for auto-credited customers; claim-based customers only get value if they file
a claim (and pay a hassle cost). 'Notify everything' policies lose retention when the customer was unaware of the failure.
"""
import numpy as np

DOSE = 4.0
REWARD_COST_RATIO = 0.35   # brand cost of fulfilling a converted-points reward, as a share of its face value
VALUE_HORIZON = 4         # a retained customer is worth 4x their 90-day margin (12-month horizon)
SILENT_F = 0.6          # a silent credit moves behaviour less than an explicit one
CLAIM_NOTICED, CLAIM_SILENT = 0.35, 0.08
HASSLE = 0.01           # retention lost per claim a customer has to file
BACKFIRE_ALL, BACKFIRE_GROUP = 0.015, 0.06


def g(c):
    return 1 - np.exp(-np.asarray(c, float) / DOSE)


def claim_p(noticed):
    return CLAIM_NOTICED if noticed else CLAIM_SILENT


def p_retain(gt, entries, arm, policy="salience"):
    """gt: dict(base_retention, true_response, backfire). entries: list of (value_usd, noticed, cost_usd)."""
    base, tau = gt["base_retention"], gt["true_response"]
    if arm == "treatment":
        c = sum(v * (1.0 if n else SILENT_F) for v, n, *_ in entries)
        p = base + tau * g(c)
        if policy == "notify_all" and any(not n for _, n, *_ in entries):
            p -= BACKFIRE_GROUP if gt["backfire"] else BACKFIRE_ALL
    else:
        cl = [(claim_p(n), v) for v, n, *_ in entries]
        p = base + tau * g(sum(cp * v for cp, v in cl)) - HASSLE * sum(cp for cp, _ in cl)
    return float(np.clip(p, 0.0, 1.0))


def entry(value, noticed, unit):
    return (value, bool(noticed), value * (REWARD_COST_RATIO if unit == "POINTS" else 1.0))


def exp_payout(entries, arm):
    """Expected BRAND COST (not customer-perceived value)."""
    return sum(c for _, _, c in entries) if arm == "treatment" else sum(claim_p(n) * c for _, n, c in entries)


def exp_contacts(entries, arm):
    return 0.0 if arm == "treatment" else sum(claim_p(n) for _, n, *_ in entries)

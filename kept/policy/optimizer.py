"""Constrained settlement optimizer.

The contract amount ("base") is ALWAYS paid - it is owed. The optimizer only allocates an optional goodwill TOP-UP:

    maximise   sum_i [ margin_rate * spend_i * dp_i(level_i) ]  -  sum_i topup_i(level_i)
    subject to sum_i topup_i <= budget,   top-up depends ONLY on ALLOWED_INPUTS (fairness)

dp_i(level) = tau_i * (g(b_i + topup) - g(b_i)) / g(b_i)   with an ASSUMED diminishing-returns dose-response
g(c) = 1 - exp(-c / DOSE_SCALE). Greedy by marginal margin-per-dollar (optimal for concave per-customer gains).
"""
import numpy as np
import pandas as pd

ALLOWED_INPUTS = ["owed_usd", "uplift_hat", "spend_hat"]      # nothing else may influence an amount
PROTECTED_LIKE = {"name", "gender", "age", "zip", "postcode", "ethnicity", "religion", "address_hash", "device_id", "payment_hash", "segment"}
DOSE_SCALE = 4.0
LEVELS = [0.0, 1 / 3, 2 / 3, 1.0]


def g(c):
    return 1 - np.exp(-np.asarray(c, float) / DOSE_SCALE)


def allocate(df: pd.DataFrame, budget: float, topup_pct: float, topup_cap: float, margin_rate: float = 0.25):
    """df columns = ALLOWED_INPUTS (+ any ids, ignored). Returns Series of top-ups aligned to df.index."""
    cols = set(df.columns) - set(ALLOWED_INPUTS)
    b = df.owed_usd.to_numpy(float); tau = df.uplift_hat.clip(lower=0).to_numpy(float); spend = df.spend_hat.to_numpy(float)
    full = np.minimum(topup_cap, topup_pct * b)
    n = len(df)
    # candidate steps (customer, level): marginal gain & marginal cost
    steps = []
    for i in range(n):
        if full[i] <= 0 or tau[i] <= 0:
            continue
        prev_gain, prev_cost = 0.0, 0.0
        for k, lv in enumerate(LEVELS[1:], 1):
            t = full[i] * lv
            gain = margin_rate * spend[i] * tau[i] * (g(b[i] + t) - g(b[i])) / max(g(b[i]), 1e-9)
            steps.append(((gain - prev_gain) / max(t - prev_cost, 1e-9), i, k, gain - prev_gain, t - prev_cost))
            prev_gain, prev_cost = gain, t
    steps.sort(reverse=True)
    level = np.zeros(n, int); spent = 0.0; pred_gain = 0.0
    for ratio, i, k, dg, dc in steps:
        if ratio <= 1.0:                    # a dollar of top-up must return more than a dollar of margin
            break
        if level[i] != k - 1 or spent + dc > budget:
            continue
        level[i] = k; spent += dc; pred_gain += dg
    top = pd.Series([np.floor(full[i] * LEVELS[level[i]] * 100) / 100 for i in range(n)], index=df.index)   # floor to cents: never exceed the budget
    top.attrs.update(spent=float(top.sum()), predicted_gain=float(pred_gain), ignored_columns=sorted(cols))
    return top


def fairness_audit(df: pd.DataFrame, topup: pd.Series, groups: pd.DataFrame, sensitive=(), tol_usd=0.25):
    """Two checks.
    1. Input gate: the optimizer may only see ALLOWED_INPUTS (no name/address/device/payment/segment...).
    2. Information test: fit a flexible model of top-up on the allowed inputs. If a group attribute (e.g. shared address,
       a proxy for household/socioeconomic status) still explains the RESIDUAL, it is an extra driver of amounts. The
       mean residual of each `sensitive` group must stay within +-tol_usd. Segment gaps are reported for information only
       (segment legitimately enters through the measured uplift, so it is not a protected attribute)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    bad_inputs = sorted(set(df.columns) & PROTECTED_LIKE)
    X = df[ALLOWED_INPUTS]
    fit = HistGradientBoostingRegressor(max_depth=4, max_iter=80, random_state=0).fit(X, topup.reindex(df.index).fillna(0.0))
    resid = topup.reindex(df.index).fillna(0.0) - fit.predict(X)
    gaps, info = [], []
    for col in groups.columns:
        for val, r in resid.groupby(groups[col]):
            rec = dict(group=col, value=str(val), n=int(len(r)), mean_topup=float(topup.reindex(r.index).mean()), mean_residual=float(r.mean()))
            (gaps if col in sensitive else info).append(rec)
    worst = max((abs(g["mean_residual"]) for g in gaps), default=0.0)
    return dict(allowed_inputs=ALLOWED_INPUTS, protected_inputs_found=bad_inputs, sensitive_groups=list(sensitive), max_abs_residual_usd=float(worst),
                tolerance_usd=tol_usd, passed=(not bad_inputs) and worst <= tol_usd, sensitive_detail=gaps, informational=info)

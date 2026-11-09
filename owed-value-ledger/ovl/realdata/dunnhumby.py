"""Real grocery data: Dunnhumby 'The Complete Journey' (2,500 households, 102 weeks, 2.6M basket lines, 30 targeted campaigns).

Two questions real data CAN answer:
  1. PRICE PROTECTION: if a shopper were refunded any price drop on the same product/store within 14 days of buying, how often
     would that happen and what would it cost?   (the 'price_protection' contract on real prices)
  2. DOES GIVING VALUE CHANGE BEHAVIOUR? Targeted coupon campaigns vs non-targeted households, difference-in-differences on
     spend and on staying active. NOT randomised (targeting was by the retailer), so it is evidence, not proof.
"""
import json
import numpy as np
import pandas as pd

from .. import config

DIR = config.DATA_DIR / "dunnhumby" / "processed"
OUT = config.DATA_DIR / "dunnhumby_metrics.json"
MARGIN = 0.25                       # assumed grocery gross margin (assumption)


def available():
    return (DIR / "transaction_data.parquet").exists()


def price_protection(t, window=14, min_pct=0.05, min_abs=0.25):
    x = t[(t.QUANTITY > 0) & (t.SALES_VALUE > 0)].copy()
    x["unit"] = x.SALES_VALUE / x.QUANTITY                                   # price actually paid per unit (after in-store discount)
    daily = x.groupby(["PRODUCT_ID", "STORE_ID", "DAY"]).unit.min()
    fut = np.full(len(x), np.inf)
    for k in range(1, window + 1):                                           # lowest price anyone paid in the next 14 days, same product + store
        idx = pd.MultiIndex.from_arrays([x.PRODUCT_ID.values, x.STORE_ID.values, x.DAY.values + k])
        v = daily.reindex(idx).to_numpy(float)
        fut = np.fmin(fut, np.where(np.isnan(v), np.inf, v))
    x["later_min"] = fut
    drop = (x.unit - x.later_min)
    hit = (drop >= min_abs) & (drop / x.unit >= min_pct) & np.isfinite(x.later_min)
    x["refund"] = np.where(hit, drop * x.QUANTITY, 0.0)
    h = x.groupby("household_key").agg(spend=("SALES_VALUE", "sum"), refund=("refund", "sum"), lines=("refund", lambda s: (s > 0).sum()))
    # how much of the drop was a deliberate promotion vs ordinary price change
    promo_after = x[hit]
    return dict(
        purchase_lines=int(len(x)), lines_with_drop=int(hit.sum()), share_lines_with_drop=float(hit.mean()),
        total_refund=float(x.refund.sum()), refund_pct_of_sales=float(x.refund.sum() / x.SALES_VALUE.sum()),
        mean_refund_per_affected_line=float(x.refund[hit].mean()), median_refund_per_affected_line=float(x.refund[hit].median()),
        households=int(len(h)), households_with_any_refund=int((h.refund > 0).sum()), share_households_affected=float((h.refund > 0).mean()),
        refund_per_household_per_year=float(h.refund.mean() / (t.DAY.max() / 365.0)),
        mean_drop_pct_when_hit=float((drop[hit] / x.unit[hit]).mean()),
        params=dict(window_days=window, min_pct=min_pct, min_abs=min_abs),
        note="'later price' = the lowest unit price ANY shopper paid for the same product in the same store in the following 14 days; includes weekly promotions, which a real price-protection policy would likely exclude or cap.")


def campaign_did(t, camp, desc, seed=0):
    """Per-campaign DiD on weekly spend and on staying active, treated = households in that campaign's list."""
    wk = t.groupby(["household_key", "WEEK_NO"]).agg(spend=("SALES_VALUE", "sum"), coupon=("COUPON_DISC", "sum")).reset_index()
    hh = np.sort(t.household_key.unique())
    spend = wk.pivot(index="household_key", columns="WEEK_NO", values="spend").reindex(hh).fillna(0.0)
    coup = wk.pivot(index="household_key", columns="WEEK_NO", values="coupon").reindex(hh).fillna(0.0)
    maxw = spend.columns.max()
    rows = []
    for c in desc.itertuples():
        s_w, e_w = int(np.ceil(c.START_DAY / 7)), int(np.ceil(c.END_DAY / 7))
        pre = range(max(1, s_w - 8), s_w)
        post = range(s_w, e_w + 1)
        after = range(e_w + 1, min(maxw, e_w + 8) + 1)
        if len(list(pre)) < 6 or len(list(post)) < 2 or e_w + 8 > maxw:
            continue
        tr = camp[camp.CAMPAIGN == c.CAMPAIGN].household_key.unique()
        T = np.isin(hh, tr)
        if T.sum() < 30 or (~T).sum() < 30:
            continue
        sp = spend[list(pre)].mean(1).to_numpy(), spend[list(post)].mean(1).to_numpy()
        cp_post = -coup[list(post)].mean(1).to_numpy()
        act = (spend[list(after)].sum(1) > 0).to_numpy().astype(float)
        act_pre = (spend[list(pre)].sum(1) > 0).to_numpy().astype(float)
        d_spend = (sp[1] - sp[0])
        q = pd.qcut(pd.Series(sp[0]).rank(method="first"), 5, labels=False).to_numpy()      # pre-period spend quintile (matching)
        rows.append(dict(campaign=int(c.CAMPAIGN), type=str(c.DESCRIPTION), weeks=len(post), n_treated=int(T.sum()), n_control=int((~T).sum()),
                         pre_spend_t=float(sp[0][T].mean()), did_weekly_spend=float(d_spend[T].mean() - d_spend[~T].mean()),
                         coupon_cost_weekly=float(cp_post[T].mean() - cp_post[~T].mean()),
                         did_active_after=float((act[T] - act_pre[T]).mean() - (act[~T] - act_pre[~T]).mean()),
                         _d=d_spend, _T=T, _cost=cp_post, _a=act - act_pre, _q=q, _q0=np.zeros_like(q)))
    # pooled estimate, bootstrap over households
    rng = np.random.default_rng(seed)
    def matched(v, T, q, w8):
        """treated-minus-control within pre-spend quintiles, weighted by the treated count in each quintile"""
        num, den = 0.0, 0.0
        for k in range(5):
            m = q == k
            a, b = v[m & T], v[m & ~T]
            if len(a) >= 3 and len(b) >= 3:
                num += (a.mean() - b.mean()) * len(a); den += len(a)
        return num / den if den else np.nan

    def pooled(idx, key="_q"):
        sp_, co_, ac_ = [], [], []
        for r in rows:
            T = r["_T"][idx]; q = r[key][idx]
            w = r["weeks"]
            sp_.append(matched(r["_d"][idx], T, q, w) * w); co_.append(matched(r["_cost"][idx], T, q, w) * w); ac_.append(matched(r["_a"][idx], T, q, w))
        return np.nanmean(sp_), np.nanmean(co_), np.nanmean(ac_)
    base = pooled(np.arange(len(hh)))
    naive = pooled(np.arange(len(hh)), "_q0")             # same estimator WITHOUT matching on pre-period spend
    naive_b = np.array([pooled(rng.integers(0, len(hh), len(hh)), "_q0") for _ in range(100)])
    nci = np.percentile(naive_b, [2.5, 97.5], axis=0)
    boots = np.array([pooled(rng.integers(0, len(hh), len(hh))) for _ in range(150)])
    ci = np.percentile(boots, [2.5, 97.5], axis=0)
    net = base[0] * MARGIN - base[1]
    net_b = boots[:, 0] * MARGIN - boots[:, 1]
    tidy = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    by_type = {}
    for r in tidy: by_type.setdefault(r["type"], []).append(r)
    return dict(
        campaigns_analysed=len(tidy),
        unmatched=dict(incremental_spend_per_targeted_household=float(naive[0]), ci95_spend=[float(nci[0][0]), float(nci[1][0])],
                       change_in_share_active_after=float(naive[2]), ci95_active=[float(nci[0][2]), float(nci[1][2])],
                       net_margin_per_targeted_household=float(naive[0] * MARGIN - naive[1]),
                       reading="WITHOUT matching on pre-period spend the sign flips: targeted households are heavier shoppers and regress to the mean. The answer depends on the adjustment, so treat both as fragile."),
        pooled=dict(
            incremental_spend_per_targeted_household=float(base[0]), ci95_spend=[float(ci[0][0]), float(ci[1][0])],
            coupon_cost_per_targeted_household=float(base[1]), ci95_cost=[float(ci[0][1]), float(ci[1][1])],
            change_in_share_active_after=float(base[2]), ci95_active=[float(ci[0][2]), float(ci[1][2])],
            net_margin_per_targeted_household=float(net), ci95_net=[float(np.percentile(net_b, 2.5)), float(np.percentile(net_b, 97.5))],
            net_positive_prob=float((net_b > 0).mean()), margin_rate_assumed=MARGIN),
        by_type=[dict(type=k, campaigns=len(v), did_weekly_spend=float(np.mean([z["did_weekly_spend"] for z in v])),
                      coupon_cost_weekly=float(np.mean([z["coupon_cost_weekly"] for z in v])),
                      did_active_after=float(np.mean([z["did_active_after"] for z in v]))) for k, v in sorted(by_type.items())],
        campaigns=tidy[:30],
        design="difference-in-differences: weekly spend in the campaign weeks vs the 8 weeks before, targeted minus non-targeted households MATCHED within pre-period-spend quintiles (targeted households are heavier shoppers, so unmatched comparisons suffer regression to the mean), pooled over campaigns; 150 household bootstraps. Targeting was NOT randomised.")


def run(verbose=True):
    if not available():
        raise FileNotFoundError(f"Dunnhumby parquet files not found in {DIR}")
    t = pd.read_parquet(DIR / "transaction_data.parquet", columns=["household_key", "PRODUCT_ID", "STORE_ID", "DAY", "QUANTITY", "SALES_VALUE", "WEEK_NO", "COUPON_DISC", "RETAIL_DISC"])
    camp = pd.read_parquet(DIR / "campaign_table.parquet")
    desc = pd.read_parquet(DIR / "campaign_desc.parquet")
    red = pd.read_parquet(DIR / "coupon_redempt.parquet")
    M = dict(dataset="Dunnhumby The Complete Journey (real grocery retail, 2,500 households, ~2 years, USD).",
             overview=dict(households=int(t.household_key.nunique()), basket_lines=int(len(t)), total_sales=float(t.SALES_VALUE.sum()), weeks=int(t.WEEK_NO.max()),
                           campaigns=int(desc.CAMPAIGN.nunique()), targeted_pairs=int(len(camp)), coupon_redemptions=int(len(red)),
                           share_sales_discounted=float((t.RETAIL_DISC < 0).mean())),
             price_protection=price_protection(t), campaigns=campaign_did(t, camp, desc),
             cannot_be_tested="late delivery, points expiry, subscriptions (absent); campaign targeting was not randomised")
    OUT.write_text(json.dumps(M, indent=1, default=float), encoding="utf8")
    if verbose: print(f"wrote {OUT}")
    return M


if __name__ == "__main__":
    import time, warnings
    warnings.filterwarnings("ignore"); t0 = time.time(); run(); print(round(time.time() - t0, 1), "s")

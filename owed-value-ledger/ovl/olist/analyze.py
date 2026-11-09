"""Real-data analysis on the Olist Brazilian e-commerce dataset (CC BY-NC-SA 4.0, ~100k orders, 2016-2018).

What real data CAN answer here (and what it cannot):
  * late-delivery promises are real (estimated vs actual date) -> the late_delivery contract runs on real delays
  * how many late customers never complained (real 'silent victims')
  * a silent-failure detector trained on REAL labels (bad review although the order was on time)
  * spike/incident discovery on real orders (validated against two well-known real events)
  * observational effect of a late first order on repeat purchase
  * review-text model (Portuguese) and complaint themes
What it CANNOT answer: price drops, points, subscriptions (not in the data) and the effect of paying a credit
(nobody was randomised). Money amounts are in BRL; credit bands below are an assumption.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score

from .. import config

DIR = config.DATA_DIR / "olist"
OUT = config.DATA_DIR / "olist_metrics.json"
DAY = 1440.0
# late-delivery bands in minutes past the promised date (Olist promises are day-granular), BRL
BANDS = [dict(lo=0, hi=1 * DAY, pct=0.05, cap=9.0), dict(lo=1 * DAY, hi=3 * DAY, pct=0.08, cap=12.0), dict(lo=3 * DAY, hi=1e9, pct=0.10, cap=15.0)]
MARGIN = 0.25


def available():
    return (DIR / "olist_orders_dataset.csv").exists()


def band_credit(delay_min, amount, bands=BANDS):
    for b in bands:
        if b["lo"] < delay_min <= b["hi"]:
            return round(min(b["cap"], amount * b["pct"]), 2)
    return 0.0


# ------------------------------------------------------------------ loading
def load():
    r = lambda f, **k: pd.read_csv(DIR / f, **k)
    o = r("olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
    c = r("olist_customers_dataset.csv")[["customer_id", "customer_unique_id", "customer_state"]]
    it = r("olist_order_items_dataset.csv")
    p = r("olist_products_dataset.csv")[["product_id", "product_category_name", "product_weight_g", "product_photos_qty", "product_description_lenght"]]
    tr = r("product_category_name_translation.csv")
    pay = r("olist_order_payments_dataset.csv")
    rv = r("olist_order_reviews_dataset.csv", parse_dates=["review_creation_date"])
    rv = rv.sort_values("review_creation_date").drop_duplicates("order_id", keep="last")
    items = it.groupby("order_id").agg(price=("price", "sum"), freight=("freight_value", "sum"), n_items=("order_item_id", "max"),
                                       seller_id=("seller_id", "first"), product_id=("product_id", "first"))
    items = items.join(p.set_index("product_id"), on="product_id").join(tr.set_index("product_category_name"), on="product_category_name")
    pays = pay.groupby("order_id").agg(installments=("payment_installments", "max"), ptype=("payment_type", lambda s: s.mode().iat[0]))
    d = o.merge(c, on="customer_id").join(items, on="order_id").join(pays, on="order_id").merge(
        rv[["order_id", "review_score", "review_comment_message", "review_creation_date"]], on="order_id", how="left")
    d = d.rename(columns={"order_purchase_timestamp": "purchase", "order_delivered_customer_date": "delivered", "order_estimated_delivery_date": "promised"})
    d["category"] = d.product_category_name_english.fillna("unknown")
    d["delay_min"] = (d.delivered - d.promised).dt.total_seconds() / 60
    d["late"] = d.delay_min > 0
    d["amount"] = d.price + d.freight
    return d


# ------------------------------------------------------------------ 1. contract on real delays
def late_contract(d):
    dl = d[(d.order_status == "delivered") & d.delivered.notna()].copy()
    dl["credit"] = [band_credit(m, a) for m, a in zip(dl.delay_min, dl.amount)]
    L = dl[dl.credit > 0]
    sev = pd.cut(L.delay_min / DAY, [0, 1, 3, 7, 1e9], labels=["<=1d", "1-3d", "3-7d", ">7d"])
    comp = L.review_score.le(2)
    wrote = L.review_comment_message.notna()
    return dl, dict(
        delivered=len(dl), late=len(L), late_rate=len(L) / len(dl), median_days_late=float((L.delay_min / DAY).median()),
        severity_mix=sev.value_counts(normalize=True).sort_index().round(3).to_dict(),
        total_credit_brl=float(L.credit.sum()), mean_credit_brl=float(L.credit.mean()), credit_pct_of_gmv=float(L.credit.sum() / dl.amount.sum()),
        silent=dict(share_with_low_review=float(comp.mean()), share_with_any_comment=float(wrote.mean()), share_no_low_review=float(1 - comp.mean()),
                    note="late customers who did NOT leave a low review: under claim-based handling their credit would be unclaimed or invisible"),
        sample=[dict(order=o[:8], delay_days=round(m / DAY, 1), amount=round(a, 2), credit=cr, wrote_low_review=bool(s <= 2) if pd.notna(s) else False)
                for o, m, a, cr, s in L.sample(8, random_state=3)[["order_id", "delay_min", "amount", "credit", "review_score"]].itertuples(index=False)])


# ------------------------------------------------------------------ 2. review score vs lateness
def review_effect(dl):
    x = dl.dropna(subset=["review_score"])
    g = x.groupby(pd.cut(x.delay_min / DAY, [-1e9, -7, 0, 1, 3, 7, 1e9], labels=["7d+ early", "early", "0-1d late", "1-3d late", "3-7d late", ">7d late"]), observed=True)
    return dict(by_delay=[dict(bucket=str(k), n=int(len(v)), mean_score=float(v.review_score.mean()), share_low=float(v.review_score.le(2).mean())) for k, v in g],
                late_vs_ontime_score_gap=float(x[~x.late].review_score.mean() - x[x.late].review_score.mean()))


# ------------------------------------------------------------------ 3. silent-failure detector on REAL labels
def seller_prior(d, lag_days=60, k=20):
    """Seller's historical bad-review rate using only orders purchased >= lag_days earlier (reviews take time), smoothed."""
    d = d.sort_values("purchase")
    bad_all = d.review_score.le(2).astype(float).where(d.review_score.notna())
    gm = float(bad_all.mean())
    out = pd.Series(gm, index=d.index)
    for sid, g in d.groupby("seller_id"):
        t = g.purchase.values.astype("datetime64[s]").astype("int64")
        b = bad_all.loc[g.index].fillna(0).values
        n = np.where(bad_all.loc[g.index].notna(), 1, 0)
        cb, cn = np.concatenate([[0], np.cumsum(b)]), np.concatenate([[0], np.cumsum(n)])
        j = np.searchsorted(t, t - lag_days * 86400, side="left")
        out.loc[g.index] = (cb[j] + k * gm) / (cn[j] + k)
    return out


FEATS = ["price", "freight", "freight_ratio", "n_items", "installments", "est_days", "ship_slack_days", "weight_g", "photos", "desc_len",
         "seller_prior_bad", "category", "state", "ptype", "purchase_month", "delay_days"]
CATS = ["category", "state", "ptype"]


def build_X(dl):
    x = pd.DataFrame(index=dl.index)
    x["price"], x["freight"] = dl.price, dl.freight
    x["freight_ratio"] = dl.freight / dl.price.clip(lower=1)
    x["n_items"], x["installments"] = dl.n_items, dl.installments
    x["est_days"] = (dl.promised - dl.purchase).dt.total_seconds() / DAY
    x["ship_slack_days"] = x.est_days                           # promised window length (kept separate name for readability)
    x["weight_g"], x["photos"], x["desc_len"] = dl.product_weight_g, dl.product_photos_qty, dl.product_description_lenght
    x["seller_prior_bad"] = seller_prior(dl).reindex(dl.index)
    x["delay_days"] = dl.delay_min / DAY                         # known at delivery time
    x["purchase_month"] = dl.purchase.dt.month
    top = dl.category.value_counts().index[:30]
    x["category"] = dl.category.where(dl.category.isin(top), "other")
    x["state"] = dl.customer_state
    x["ptype"] = dl.ptype.fillna("unknown")
    for c in CATS:
        x[c] = x[c].astype("category").cat.codes
    return x[FEATS].drop(columns=["ship_slack_days"], errors="ignore")


def silent_failure(dl):
    """Label: bad review (<=2) although the order was NOT late = a failure with no delivery telemetry."""
    x = dl[(~dl.late) & dl.review_score.notna()].copy()
    X = build_X(dl).loc[x.index]
    X = X.drop(columns=["delay_days"]).assign(early_days=-(x.delay_min / DAY))
    y = x.review_score.le(2).astype(int)
    tr = x.purchase < "2018-02-01"; ca = (x.purchase >= "2018-02-01") & (x.purchase < "2018-04-01"); te = x.purchase >= "2018-04-01"
    cat_idx = [list(X.columns).index(c) for c in CATS]
    gbm = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.06, l2_regularization=2.0, categorical_features=cat_idx, random_state=0)
    gbm.fit(X[tr], y[tr])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(gbm.predict_proba(X[ca])[:, 1], y[ca])
    raw = gbm.predict_proba(X[te])[:, 1]; cal = iso.predict(raw); yt = y[te]
    thr = float(np.quantile(cal, 0.9))                              # flag the riskiest 10%
    bins = np.linspace(0, cal.max() + 1e-9, 9); idx = np.clip(np.digitize(cal, bins) - 1, 0, 7)
    rel = [dict(mean_pred=float(cal[idx == b].mean()), observed=float(yt.to_numpy()[idx == b].mean()), n=int((idx == b).sum())) for b in range(8) if (idx == b).sum() >= 30]
    pi = permutation_importance(gbm, X[te].iloc[:6000], yt.iloc[:6000], n_repeats=3, random_state=0, scoring="roc_auc")
    flag = cal >= thr
    return dict(task="bad review although delivered on time (a failure invisible to delivery telemetry)", n_train=int(tr.sum()), n_test=int(te.sum()),
                base_rate_test=float(yt.mean()), split="train < 2018-02, calibrate Feb-Mar 2018, test Apr 2018+ (temporal)",
                auc=float(roc_auc_score(yt, cal)), brier_raw=float(brier_score_loss(yt, raw)), brier_calibrated=float(brier_score_loss(yt, cal)),
                top10pct_precision=float(precision_score(yt, flag)), top10pct_recall=float(recall_score(yt, flag)), lift_top10pct=float(precision_score(yt, flag) / yt.mean()),
                reliability=rel, permutation_importance=sorted([dict(feature=f, importance=float(v)) for f, v in zip(X.columns, pi.importances_mean)], key=lambda z: -z["importance"])[:8])


# ------------------------------------------------------------------ 4. spike / incident discovery on real data
KNOWN = {"2017-11-24": "Black Friday 2017 (external knowledge: known surge in delivery delays)", "2018-05-28": "Brazilian truckers' strike, late May 2018 (external knowledge)"}


def spikes(dl):
    """Daily late-rate spikes by purchase date vs a robust 91-day rolling-median baseline (Poisson tail), merged into windows."""
    x = dl[(dl.purchase >= "2017-01-01") & (dl.purchase <= "2018-08-01")]
    g = x.groupby(x.purchase.dt.floor("D")).agg(n=("late", "size"), k=("late", "sum"))
    g["rate"] = g.k / g.n
    g["base"] = g.rate.rolling(91, center=True, min_periods=30).median()      # robust to the spike itself
    g["p"] = [poisson.sf(k - 1, max(1e-9, b * n)) if n >= 30 else 1.0 for k, n, b in zip(g.k, g.n, g.base)]
    hit = g[(g.p < 1e-6) & (g.k >= 15)]
    g = g.assign(p=g.p)
    wins, cur = [], None
    for dt, r in hit.iterrows():
        if cur and (dt - cur["end"]).days <= 3:
            cur["end"] = dt; cur["days"].append(r)
        else:
            if cur: wins.append(cur)
            cur = dict(start=dt, end=dt, days=[r])
    if cur: wins.append(cur)
    out = []
    for w in wins:
        dd = pd.DataFrame(w["days"])
        inw = x[(x.purchase >= w["start"]) & (x.purchase < w["end"] + pd.Timedelta(days=1)) & x.late]
        match = [v for k, v in KNOWN.items() if abs((pd.Timestamp(k) - w["start"]).days) <= 10 or (w["start"] <= pd.Timestamp(k) <= w["end"] + pd.Timedelta(days=10))]
        out.append(dict(start=str(w["start"].date()), end=str(w["end"].date()), days=len(dd), late_rate=float(dd.k.sum() / dd.n.sum()), baseline=float(dd.base.mean()),
                        late_orders=int(len(inw)), customers=int(inw.customer_unique_id.nunique()),
                        silent_victims=int(inw[inw.review_score.fillna(5) > 2].customer_unique_id.nunique()),
                        min_p=float(max(dd.p.min(), 1e-300)), matches_known_event=match[0] if match else None))
    return sorted(out, key=lambda z: -z["late_orders"])[:8]


def seller_incidents(dl):
    """Seller-weeks whose bad-review count is surprising given the seller's own base rate."""
    x = dl.dropna(subset=["review_score"]).copy()
    x["bad"] = x.review_score.le(2)
    x["wk"] = x.purchase.dt.to_period("W").dt.start_time
    s = x.groupby("seller_id").agg(N=("bad", "size"), K=("bad", "sum"))
    w = x.groupby(["seller_id", "wk"]).agg(n=("bad", "size"), k=("bad", "sum")).reset_index().join(s, on="seller_id")
    w["p0"] = ((w.K - w.k) / (w.N - w.n).clip(lower=1)).clip(lower=0.02)
    w["p"] = [poisson.sf(k - 1, p0 * n) if (n >= 8 and N - n >= 30) else 1.0 for k, n, p0, N in zip(w.k, w.n, w.p0, w.N)]
    hit = w[(w.p < 1e-5) & (w.k >= 5)].sort_values("p")
    return dict(tested=int(((w.n >= 8) & (w.N - w.n >= 30)).sum()), flagged=int(len(hit)), expected_false=float(((w.n >= 8) & (w.N - w.n >= 30)).sum() * 1e-5),
                top=[dict(seller=str(r.seller_id)[:8], week=str(r.wk.date()), orders=int(r.n), bad=int(r.k), seller_base_rate=float(r.p0), p=float(r.p)) for r in hit.head(6).itertuples()])


# ------------------------------------------------------------------ 5. review-text model (Portuguese)
THEMES = {"delay": r"atras|demor|prazo|n[aã]o chegou|ainda n[aã]o|esperando", "not_received": r"n[aã]o recebi|n[aã]o chegou|nunca chegou|n[aã]o entreg",
          "damaged_defective": r"quebrad|defeit|danific|estragad|n[aã]o funciona|amassad", "wrong_item": r"errad|diferente|outro produto|n[aã]o era",
          "quality": r"qualidade|p[eé]ssim|horr[ií]vel|ruim|fraco"}


def text_model(dl):
    x = dl[dl.review_comment_message.notna() & dl.review_score.notna()].copy()
    x["bad"] = x.review_score.le(2).astype(int)
    tr = x.purchase < "2018-04-01"
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, max_features=60000)
    Xtr, Xte = vec.fit_transform(x[tr].review_comment_message), vec.transform(x[~tr].review_comment_message)
    lr = LogisticRegression(max_iter=1000, C=3).fit(Xtr, x[tr].bad)
    auc = float(roc_auc_score(x[~tr].bad, lr.predict_proba(Xte)[:, 1]))
    names = np.array(vec.get_feature_names_out()); co = lr.coef_[0]
    badx = x[x.bad == 1]
    themes = {k: float(badx.review_comment_message.str.contains(rx, case=False, regex=True).mean()) for k, rx in THEMES.items()}
    lateb = badx[badx.late.fillna(False)]
    mention = float(lateb.review_comment_message.str.contains(THEMES["delay"] + "|" + THEMES["not_received"], case=False, regex=True).mean()) if len(lateb) else None
    return dict(n_with_comment=int(len(x)), auc_bad_review_from_text=auc, split="train < 2018-04, test after",
                top_bad_terms=names[np.argsort(-co)[:12]].tolist(), top_good_terms=names[np.argsort(co)[:8]].tolist(),
                bad_review_themes=themes, share_of_late_low_reviews_mentioning_delay=mention,
                note="Portuguese reviews; themes by keyword lexicon (no ground-truth labels exist in Olist)")


# ------------------------------------------------------------------ 6. does a late FIRST order reduce repeat purchase?
def repeat_effect(d, dl, seed=0):
    cu = d.sort_values("purchase")
    first = cu.drop_duplicates("customer_unique_id", keep="first")
    last_seen = d.purchase.max()
    nxt = d.sort_values("purchase").groupby("customer_unique_id").purchase.apply(lambda s: s.iloc[1] if len(s) > 1 else pd.NaT)
    f = first.set_index("customer_unique_id").join(nxt.rename("second"))
    f = f[(f.order_status == "delivered") & f.delivered.notna() & (f.purchase <= last_seen - pd.Timedelta(days=210))].copy()
    f["repeat"] = ((f.second - f.purchase).dt.days <= 180).astype(int)         # a different order within 180 days
    f["T"] = f.late.astype(int)
    X = build_X(f.reset_index(drop=True).set_index(f.index)).drop(columns=["delay_days", "seller_prior_bad"], errors="ignore") if False else None
    # confounders known BEFORE delivery outcome (price, freight, promise length, category, state, payment, month)
    Z = pd.DataFrame({"price": f.price, "freight": f.freight, "freight_ratio": f.freight / f.price.clip(lower=1), "n_items": f.n_items, "installments": f.installments,
                      "est_days": (f.promised - f.purchase).dt.total_seconds() / DAY, "weight": f.product_weight_g, "month": f.purchase.dt.month,
                      "cat": f.category.astype("category").cat.codes, "state": f.customer_state.astype("category").cat.codes, "pay": f.ptype.astype("category").cat.codes})
    Z = Z.fillna(Z.median())
    t, y = f["T"].to_numpy(), f["repeat"].to_numpy()
    prop = HistGradientBoostingClassifier(max_iter=80, max_depth=3, learning_rate=0.06, random_state=0)
    from sklearn.model_selection import cross_val_predict
    e = np.clip(cross_val_predict(prop, Z, t, cv=5, method="predict_proba")[:, 1], 0.01, 0.5)
    def ipw(idx):
        w1 = t[idx] / e[idx]; w0 = (1 - t[idx]) / (1 - e[idx])
        return float((w1 * y[idx]).sum() / w1.sum() - (w0 * y[idx]).sum() / w0.sum())
    rng = np.random.default_rng(seed); n = len(f)
    boots = [ipw(rng.integers(0, n, n)) for _ in range(200)]
    raw = float(y[t == 1].mean() - y[t == 0].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    by = []
    sev = pd.cut(f.delay_min / DAY, [-1e9, 0, 3, 7, 1e9], labels=["on time", "late 0-3d", "late 3-7d", "late >7d"])
    for k, g in f.groupby(sev, observed=True): by.append(dict(bucket=str(k), n=int(len(g)), repeat_rate=float(g["repeat"].mean())))
    lowr = f.review_score.le(2)
    return dict(n_customers=int(n), n_late=int(t.sum()), repeat_rate_on_time=float(y[t == 0].mean()), repeat_rate_late=float(y[t == 1].mean()),
                raw_difference_pp=raw * 100, adjusted_effect_pp=ipw(np.arange(n)) * 100, ci95_pp=[float(lo * 100), float(hi * 100)],
                significant=bool(lo > 0 or hi < 0), by_severity=by,
                repeat_rate_low_review=float(f[lowr]["repeat"].mean()), repeat_rate_good_review=float(f[~lowr & f.review_score.notna()]["repeat"].mean()),
                design="first order per customer, delivered, >=210 days of follow-up; outcome = another order within 180 days; IPW with cross-fitted GBM propensity; 200 bootstraps. OBSERVATIONAL (not randomised).")


def run(verbose=True):
    if not available():
        raise FileNotFoundError(f"Olist CSVs not found in {DIR}")
    d = load()
    dl, contract = late_contract(d)
    rep = repeat_effect(d, dl)
    # economics: what is the retention benefit of removing one late delivery, vs the credit it would cost?
    avg_order = float(d.amount.mean())
    late_loss_pp = max(0.0, -rep["adjusted_effect_pp"]) / 100
    benefit = late_loss_pp * avg_order * MARGIN
    econ = dict(avg_order_brl=avg_order, margin_rate=MARGIN, repeat_loss_per_late_order_pp=late_loss_pp * 100, retention_benefit_per_late_order_brl=benefit,
                mean_credit_per_late_order_brl=contract["mean_credit_brl"],
                credit_covered_by_retention_ratio=float(benefit / contract["mean_credit_brl"]) if contract["mean_credit_brl"] else None,
                reading="Even if a credit fully undid the repeat-purchase loss caused by lateness, the retention benefit is only this fraction of the credit's cost. The case for automatic credits rests on trust, reviews and fairness, not on repeat purchases.")
    M = dict(dataset="Olist Brazilian E-Commerce (real, 2016-2018). Currency BRL. Observational; no randomised credits exist in this data.",
             late_contract=contract, review_effect=review_effect(dl), silent_failure=silent_failure(dl), spikes=spikes(dl), seller_incidents=seller_incidents(dl),
             text_model=text_model(dl), repeat_effect=rep, economics=econ,
             cannot_be_tested="price protection, points, subscriptions (absent); effect of paying a credit (never randomised); uplift model")
    OUT.write_text(json.dumps(M, indent=1, default=float), encoding="utf8")
    if verbose: print(f"wrote {OUT}")
    return M


if __name__ == "__main__":
    import warnings, time
    warnings.filterwarnings("ignore"); t0 = time.time(); run(); print(round(time.time() - t0, 1), "s")

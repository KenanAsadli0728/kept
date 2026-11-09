"""Real customer-support conversations: 'Customer Support on Twitter' (2.8M tweets, ~100 brands, mostly Oct-Dec 2017).

What real data answers: how much EFFORT customers face (unanswered tweets, waiting time, repeating themselves), what they
complain about, and whether incident (spike) discovery finds real outages. It has no orders, so it cannot say who was owed what.
"""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.feature_extraction.text import TfidfVectorizer

from .. import config

CSV = config.DATA_DIR / "twcs" / "twcs" / "twcs.csv"
OUT = config.DATA_DIR / "twcs_metrics.json"

THEMES = {
    "delay_or_wait": r"\b(late|delay|delayed|waiting|wait|still no|hours|days|slow)\b",
    "outage_or_not_working": r"\b(down|outage|not working|doesn'?t work|won'?t (load|work|open)|crash|error|broken|bug|can'?t (log|connect|open))\b",
    "refund_or_money": r"\b(refund|charged|charge|billing|bill|money|credit|overcharged|payment|fee)\b",
    "lost_or_missing": r"\b(lost|missing|never (arrived|received|got)|didn'?t (arrive|receive))\b",
    "no_response": r"\b(no (one|response|reply|answer)|ignored|nobody|unresponsive|still waiting)\b",
    "account_access": r"\b(account|password|login|log in|locked|sign in|verify)\b",
}
SECTOR = {}


def available():
    return CSV.exists()


def load():
    d = pd.read_csv(CSV, usecols=["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"],
                    dtype={"author_id": str, "response_tweet_id": str})
    d["created_at"] = pd.to_datetime(d.created_at, format="%a %b %d %H:%M:%S %z %Y", utc=True).dt.tz_localize(None)
    d["inbound"] = d.inbound.astype(bool)
    return d


def threads(d):
    """Root every tweet via pointer jumping on in_response_to; returns d with root, plus per-thread stats."""
    ids = d.tweet_id.to_numpy()
    pos = pd.Series(np.arange(len(d)), index=ids)
    par = d.in_response_to_tweet_id.map(pos).to_numpy()
    par = np.where(np.isnan(par), np.arange(len(d)), par).astype(np.int64)
    for _ in range(12):                                 # pointer jumping converges quickly (threads are short)
        nxt = par[par]
        if (nxt == par).all(): break
        par = nxt
    d = d.assign(root=par)
    return d


def run(verbose=True):
    if not available():
        raise FileNotFoundError(f"Twitter support CSV not found: {CSV}")
    d = load()
    brands = set(d.loc[~d.inbound, "author_id"])
    d = threads(d)
    start = d[d.inbound & d.in_response_to_tweet_id.isna()].copy()
    start["brand"] = start.text.str.extract(r"^@(\w+)", expand=False)
    start = start[start.brand.isin(brands)]
    # first brand reply time
    created = d.set_index("tweet_id").created_at
    first_reply = start.response_tweet_id.str.split(",").str[0].astype(float)
    ok = first_reply.isin(created.index)
    rt = pd.Series(np.nan, index=start.index)
    rep_time = created.reindex(first_reply[ok]).to_numpy()
    is_brand_reply = d.set_index("tweet_id").inbound.reindex(first_reply[ok]).to_numpy() == False
    rt.loc[ok[ok].index] = np.where(is_brand_reply, (rep_time - start.loc[ok[ok].index, "created_at"].to_numpy()) / np.timedelta64(1, "m"), np.nan)
    start["reply_min"] = rt
    start["replied"] = start.reply_min.notna()
    # thread effort: how many messages did the customer send in the thread
    th = d.groupby("root").agg(cust_msgs=("inbound", "sum"), brand_msgs=("inbound", lambda s: (~s).sum()), n=("inbound", "size"))
    start["root"] = d.loc[start.index, "root"].to_numpy()
    start = start.join(th, on="root")
    # re-contact: same customer, same brand, another thread within 7 days
    s2 = start.sort_values("created_at")
    nxt = s2.groupby(["author_id", "brand"]).created_at.shift(-1)
    s2["recontact7d"] = ((nxt - s2.created_at) <= pd.Timedelta(days=7))
    start["recontact7d"] = s2.recontact7d.reindex(start.index).fillna(False)

    # per-brand table
    g = start.groupby("brand").agg(threads=("replied", "size"), reply_rate=("replied", "mean"), median_reply_min=("reply_min", "median"),
                                   p90_reply_min=("reply_min", lambda s: s.quantile(0.9)), recontact=("recontact7d", "mean"), cust_msgs=("cust_msgs", "mean"))
    g = g[g.threads >= 2000].sort_values("reply_rate")
    wait_effect = []
    r = start[start.replied]
    for lo, hi, lab in [(0, 15, "<15 min"), (15, 60, "15-60 min"), (60, 240, "1-4 h"), (240, 1e9, ">4 h")]:
        x = r[(r.reply_min >= lo) & (r.reply_min < hi)]
        wait_effect.append(dict(wait=lab, n=int(len(x)), recontact_7d=float(x.recontact7d.mean()), customer_msgs=float(x.cust_msgs.mean())))
    text = start.text.str.lower()
    themes = {k: float(text.str.contains(rx, regex=True).mean()) for k, rx in THEMES.items()}

    # spikes: brand-days of inbound volume vs the brand's robust daily baseline
    start["day"] = start.created_at.dt.floor("D")
    cnt = start.groupby(["brand", "day"]).size().rename("k").reset_index()
    big = cnt.groupby("brand").k.sum()
    cnt = cnt[cnt.brand.isin(big[big >= 3000].index)]
    med = cnt.groupby("brand").k.transform("median")
    cnt["base"] = med.clip(lower=3)
    cnt["p"] = [poisson.sf(k - 1, b) for k, b in zip(cnt.k, cnt.base)]
    ntest = len(cnt)
    sp = cnt[(cnt.p < 1e-12) & (cnt.k >= 3 * cnt.base) & (cnt.k >= 150)].sort_values("p")
    # merge consecutive days per brand and describe each with contrast terms
    inc = []
    for b, gb in sp.groupby("brand"):
        days = sorted(gb.day)
        win = [(days[0], days[0])]
        for x in days[1:]:
            if (x - win[-1][1]).days <= 1: win[-1] = (win[-1][0], x)
            else: win.append((x, x))
        for a, z in win:
            inwin = start[(start.brand == b) & (start.day >= a) & (start.day <= z)]
            rest = start[(start.brand == b) & ((start.day < a - pd.Timedelta(days=1)) | (start.day > z + pd.Timedelta(days=1)))]
            rest = rest.sample(min(8000, len(rest)), random_state=0)
            if len(rest) < 50:
                continue
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=5, max_features=20000, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
            m = vec.fit_transform(pd.concat([inwin.text, rest.text]).str.replace(r"@\w+|https?://\S+", " ", regex=True))
            diff = np.asarray(m[:len(inwin)].mean(0)).ravel() - np.asarray(m[len(inwin):].mean(0)).ravel()
            terms = np.array(vec.get_feature_names_out())[np.argsort(-diff)[:6]].tolist()
            base = float(gb.base.iloc[0])
            inc.append(dict(brand=b, start=str(a.date()), end=str(z.date()), tweets=int(len(inwin)), customers=int(inwin.author_id.nunique()),
                            baseline_per_day=base, peak_x_baseline=float(gb.k.max() / base), top_terms=terms,
                            no_reply_share=float(1 - inwin.replied.mean()), example=re.sub(r"\s+", " ", inwin.text.iloc[0])[:140]))
    inc = sorted(inc, key=lambda z: -z["tweets"])[:10]

    M = dict(dataset="Customer Support on Twitter (real, ~2.8M tweets; brands answer customers on Twitter). Mostly Oct-Dec 2017.",
             overview=dict(tweets=int(len(d)), customer_threads=int(len(start)), brands=int(start.brand.nunique()), customers=int(start.author_id.nunique()),
                           first_day=str(d.created_at.min().date()), last_day=str(d.created_at.max().date())),
             effort=dict(no_reply_share=float(1 - start.replied.mean()), median_reply_min=float(start.reply_min.median()), p90_reply_min=float(start.reply_min.quantile(0.9)),
                         recontact_7d=float(start.recontact7d.mean()), customer_msgs_mean=float(start.cust_msgs.mean()),
                         share_threads_3plus_customer_msgs=float((start.cust_msgs >= 3).mean()), by_wait=wait_effect),
             themes=themes,
             worst_reply_rate=[dict(brand=i, threads=int(r_.threads), reply_rate=float(r_.reply_rate), median_reply_min=None if pd.isna(r_.median_reply_min) else float(r_.median_reply_min)) for i, r_ in g.head(6).iterrows()],
             best_reply_rate=[dict(brand=i, threads=int(r_.threads), reply_rate=float(r_.reply_rate), median_reply_min=None if pd.isna(r_.median_reply_min) else float(r_.median_reply_min)) for i, r_ in g.tail(5).iloc[::-1].iterrows()],
             spikes=dict(brand_days_tested=int(ntest), expected_false=float(ntest * 1e-12), found=len(inc), incidents=inc),
             cannot_be_tested="who was owed what (no orders), credits, retention (no customer outcomes)")
    OUT.write_text(json.dumps(M, indent=1, default=float), encoding="utf8")
    if verbose: print(f"wrote {OUT}")
    return M


if __name__ == "__main__":
    import time, warnings
    warnings.filterwarnings("ignore"); t0 = time.time(); run(); print(round(time.time() - t0, 1), "s")

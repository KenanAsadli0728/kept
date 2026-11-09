"""Embedding-based incident discovery.

1. Keep complaints that describe a failure (per the NLP classifier; 'other'/'price'/'refund_status' are not incidents).
2. Embed text (sentence-transformers MiniLM if OVL_EMBED=st and available, else TF-IDF + SVD).
3. DBSCAN over [text embedding, scaled order date] -> clusters = candidate incidents (no predefined label).
4. Validate each cluster against operations: complaints must concentrate in one (warehouse|carrier, date) group.
5. A candidate must be statistically surprising (Poisson tail vs the base complaint rate, Bonferroni-strict).
6. Every order in that group is a victim; non-complainers become 'silent victims' and get an IOU too.
"""
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .. import config

FAILURE_TYPES = {"late", "damaged", "spoiled", "not_received"}


def embed(texts, backend=None):
    """Returns (matrix, backend_used)."""
    backend = backend or config.EMBED_BACKEND
    if backend == "st":
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer("all-MiniLM-L6-v2")
            return normalize(m.encode(list(texts), show_progress_bar=False)), "all-MiniLM-L6-v2"
        except Exception:
            pass
    tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    k = max(2, min(30, tf.shape[1] - 1, tf.shape[0] - 1))
    return normalize(TruncatedSVD(k, random_state=0).fit_transform(tf)), "tfidf+svd"


def discover(contacts: pd.DataFrame, orders: pd.DataFrame, nlp_model, params, now, eps=0.85, day_scale=0.5, backend=None):
    """contacts: id, customer_id, created_at, text, order_id. orders: id, customer_id, placed_at, carrier, warehouse, status.
    Returns list of incident dicts."""
    horizon = now - pd.Timedelta(days=params["lookback_days"])
    c = contacts[contacts.created_at >= horizon].copy()
    if c.empty:
        return []
    c["ctype"] = nlp_model.predict(c.text.tolist())
    c = c[c.ctype.isin(FAILURE_TYPES) & c.order_id.notna()].merge(
        orders[["id", "placed_at", "carrier", "warehouse"]].rename(columns={"id": "order_id"}), on="order_id")
    if len(c) < params["min_complaints"]:
        return []
    E, used = embed(c.text.tolist(), backend)
    day = ((c.placed_at - c.placed_at.min()).dt.total_seconds() / 86400).to_numpy()[:, None] * day_scale
    labels = DBSCAN(eps=eps, min_samples=3).fit_predict(np.hstack([E, day]))
    c["cluster"] = labels
    o = orders.copy()
    o["pdate"] = o.placed_at.dt.date
    c["pdate"] = c.placed_at.dt.date
    # per text-cluster: which operational key (warehouse|carrier, date) do its complaints concentrate in?
    merged = {}
    for cl, g in c[c.cluster >= 0].groupby("cluster"):
        cand = []
        for dim in ("warehouse", "carrier"):
            (val, d), n = Counter(zip(g[dim], g.pdate)).most_common(1)[0]
            cand.append((n / len(g), dim, val, d))
        conc, dim, val, d = max(cand, key=lambda t: t[0])
        if conc < 0.6:
            continue
        merged.setdefault((dim, val, d), []).append(g)
    n_orders = max(1, int((o.placed_at >= horizon).sum()))
    out = []
    for (dim, val, d), gs in merged.items():     # text clusters of the same incident (different wording) are merged by key
        g = pd.concat(gs)
        lo, hi = g.placed_at.min() - pd.Timedelta(hours=2), g.placed_at.max() + pd.Timedelta(hours=2)
        grp = o[(o[dim] == val) & (o.pdate == d) & (o.status != "shipped") & o.placed_at.between(lo, hi)]   # incident window, not the whole day
        n = int(g.customer_id.nunique())
        share = n / max(1, len(grp))
        ftype = Counter(g.ctype).most_common(1)[0][0]
        # base rate of THIS kind of complaint per order, estimated outside the candidate group (robust to the incident itself)
        rate = max(5e-4, ((c.ctype == ftype).sum() - len(g)) / max(1, n_orders - len(grp)))
        pval = float(poisson.sf(n - 1, rate * len(grp)))      # is this many complaints surprising for a group this size?
        if n < params["min_complaints"] or pval > params["max_p_value"]:
            continue
        complainers = set(g.customer_id)
        out.append(dict(
            id=f"INC-{dim[:3].upper()}-{val}-{d:%m%d}", dimension=dim, value=val, date=str(d),
            failure_type=Counter(g.ctype).most_common(1)[0][0], n_complaints=int(len(g)), complaint_customers=sorted(int(x) for x in complainers),
            victim_orders=[int(x) for x in grp.id], victim_customers=sorted(int(x) for x in set(grp.customer_id)),
            n_affected=int(grp.customer_id.nunique()), n_complained=int(len(complainers & set(grp.customer_id))),
            n_silent=int(len(set(grp.customer_id) - complainers)), p_value=pval, group_share=share, embed_backend=used))
    return sorted(out, key=lambda x: -x["n_affected"])

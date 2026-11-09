"""Points models.

1. RewardModel   - which reward will this member pick? (logistic regression on past-redemption shares)
2. SurvivalModel - will this member redeem BEFORE expiry? Discrete-time survival: rows are (member, horizon h), label =
   'redeemed within h days', right-censored cycles contribute only the horizons they were observed for. P(not redeem
   before expiry) = 1 - P(T <= days_left). Auto-conversion only fires above a probability threshold.
Features come from operational history only (never `_ground_truth`).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

HORIZONS = [3, 7, 14, 30, 60, 90]
SEG = {"new": 0, "regular": 1, "vip": 2}


class RewardModel:
    def __init__(self, rewards):
        self.rewards = list(rewards)
        self.ix = {r: i for i, r in enumerate(self.rewards)}
        self.clf = None
        self.top = self.rewards[0]
        self.report = {}

    def _shares(self, names):
        v = np.zeros(len(self.rewards))
        for n in names:
            if n in self.ix: v[self.ix[n]] += 1
        return v / max(1, v.sum())

    def fit(self, redemptions: pd.DataFrame):
        r = redemptions.sort_values("redeemed_at")
        self.top = r.reward_name.value_counts().index[0]
        X, y = [], []
        for cid, g in r.groupby("customer_id"):
            names = g.reward_name.tolist()
            if len(names) >= 3:
                X.append(self._shares(names[:-1])); y.append(names[-1])
        X, y = np.array(X), np.array(y)
        cut = int(len(X) * 0.75)
        self.clf = LogisticRegression(max_iter=500, C=3).fit(X[:cut], y[:cut])
        naive = np.array([self.rewards[int(np.argmax(x))] for x in X[cut:]])
        self.report = dict(n_train=cut, n_test=len(X) - cut, accuracy=float((self.clf.predict(X[cut:]) == y[cut:]).mean()),
                           baseline_most_frequent_accuracy=float((naive == y[cut:]).mean()), global_most_popular_accuracy=float((y[cut:] == self.top).mean()))
        self.clf.fit(X, y)
        return self

    def predict(self, names):
        if not names: return self.top
        return str(self.clf.predict([self._shares(names)])[0])


def member_features(cust: pd.DataFrame, reds: pd.DataFrame, hist: pd.DataFrame, points: pd.DataFrame, now):
    g = reds.groupby("customer_id").redeemed_at.agg(["count", "max"])
    h = hist.groupby("customer_id").redeemed.mean().rename("prior_rate")
    df = cust.set_index("id")[["segment", "created_at"]].join(g).join(h).join(points.set_index("customer_id")[["balance", "expires_at"]])
    df["n_red"] = df["count"].fillna(0)
    df["days_since_last"] = ((now - df["max"]).dt.days).fillna(400).clip(upper=400)
    df["tenure"] = (now - pd.to_datetime(df.created_at)).dt.days
    df["seg"] = df.segment.map(SEG)
    df["prior_rate"] = df.prior_rate.fillna(0.3)
    df["days_left"] = (df.expires_at - now).dt.total_seconds() / 86400
    return df


SFEATS = ["h", "balance", "n_red", "days_since_last", "prior_rate", "tenure", "seg"]


class SurvivalModel:
    def __init__(self):
        self.gbm = HistGradientBoostingClassifier(max_iter=80, max_depth=3, learning_rate=0.1, random_state=0)
        self.report = {}

    def fit(self, hist: pd.DataFrame, feats: pd.DataFrame):
        # prior_rate for a history row = redemption rate of the member's OTHER cycle (no self-leakage)
        cyc = hist.pivot(index="customer_id", columns="cycle", values="redeemed")
        other = {}
        for cid, row in cyc.iterrows():
            other[(cid, 0)] = row.get(1, np.nan); other[(cid, 1)] = row.get(0, np.nan)
        hist = hist.copy()
        hist["other"] = [other.get((c, k), 0.3) for c, k in zip(hist.customer_id, hist.cycle)]
        f = feats[["n_red", "days_since_last", "tenure", "seg"]]
        rows = []
        for r in hist.join(f, on="customer_id").itertuples(index=False):
            for hz in HORIZONS:
                ev = bool(r.redeemed) and r.days_to_redeem <= hz
                if ev or r.window_days >= hz:
                    rows.append((hz, r.balance, r.n_red, r.days_since_last, r.other, r.tenure, r.seg, int(ev), r.customer_id))
        d = pd.DataFrame(rows, columns=SFEATS + ["y", "customer_id"])
        ids = d.customer_id.unique()
        rng = np.random.default_rng(0)
        te = set(rng.choice(ids, size=len(ids) // 4, replace=False))
        m = d.customer_id.isin(te)
        self.gbm.fit(d[~m][SFEATS], d[~m].y)
        p = self.gbm.predict_proba(d[m][SFEATS])[:, 1]
        self.report = dict(n_rows=len(d), auc_within_horizon=float(roc_auc_score(d[m].y, p)), brier=float(brier_score_loss(d[m].y, p)),
                           note="discrete-time survival: classifier over (member, horizon) rows with right-censoring")
        self.gbm.fit(d[SFEATS], d.y)
        return self

    def p_not_redeem(self, feats: pd.DataFrame):
        """P(no redemption before expiry) for each row of `feats` (needs days_left)."""
        X = pd.DataFrame({"h": feats.days_left.clip(lower=0.5, upper=90), "balance": feats.balance, "n_red": feats.n_red,
                          "days_since_last": feats.days_since_last, "prior_rate": feats.prior_rate, "tenure": feats.tenure, "seg": feats.seg})
        return 1 - self.gbm.predict_proba(X[SFEATS])[:, 1]

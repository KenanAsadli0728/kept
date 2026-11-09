"""Silent-failure detector: predicts 'likely failed order' (damaged etc.) with no explicit telemetry.

Gradient boosting + isotonic calibration (so a score of 0.7 means ~70% of such orders really failed).
Explanations: occlusion attribution (replace one feature by its reference value, measure the probability drop) -
a cheap local analogue of SHAP that needs no extra dependency.
Features are built from operational tables ONLY (never `_ground_truth`).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split

CATS = ["groceries", "electronics", "apparel", "home", "beauty", "glassware"]
CARRIERS = ["A", "B", "C"]
FEATURES = ["delay_min", "amount", "tracking_views", "returned", "contact_3d", "cust_return_rate", "cust_n_orders", "category", "carrier"]
CATEGORICAL = [FEATURES.index("category"), FEATURES.index("carrier")]


def build_features(orders: pd.DataFrame, contacts: pd.DataFrame) -> pd.DataFrame:
    o = orders[orders.delivered_at.notna()].copy()
    o["delay_min"] = (o.delivered_at - o.promised_at).dt.total_seconds() / 60
    o["returned"] = (o.status == "returned").astype(int)
    g = orders.assign(r=(orders.status == "returned").astype(int)).groupby("customer_id").r.agg(["sum", "count"])
    o = o.join(g, on="customer_id")
    o["cust_n_orders"] = o["count"]
    o["cust_return_rate"] = ((o["sum"] - o.returned) / (o["count"] - 1).clip(lower=1))
    # contact within 3 days after delivery (any contact by that customer, or one explicitly tied to the order)
    c = contacts[["customer_id", "created_at", "order_id"]]
    tied = set(c.order_id.dropna().astype(int))
    m = o[["id", "customer_id", "delivered_at"]].merge(c, on="customer_id", how="left")
    near = (m.created_at >= m.delivered_at) & (m.created_at <= m.delivered_at + pd.Timedelta(days=3))
    hit = set(m.loc[near, "id"])
    o["contact_3d"] = [int(i in hit or i in tied) for i in o.id]
    o["category"] = o.category.map({c_: i for i, c_ in enumerate(CATS)}).fillna(0).astype(int)
    o["carrier"] = o.carrier.map({c_: i for i, c_ in enumerate(CARRIERS)}).fillna(0).astype(int)
    return o.set_index("id")[FEATURES + ["customer_id"]]


def describe(feature, value):
    v = value
    return {
        "delay_min": lambda: f"delivered {v:.0f} min vs promise",
        "amount": lambda: f"order value ${v:.0f}",
        "tracking_views": lambda: f"{int(v)} tracking checks",
        "returned": lambda: "order was returned" if v else "not returned",
        "contact_3d": lambda: "support contact within 3 days" if v else "no support contact",
        "cust_return_rate": lambda: f"customer return rate {v:.0%}",
        "cust_n_orders": lambda: f"{int(v)} orders on account",
        "category": lambda: f"category {CATS[int(v)]}",
        "carrier": lambda: f"carrier {CARRIERS[int(v)]}",
    }[feature]()


class SilentFailureModel:
    def __init__(self):
        self.gbm = HistGradientBoostingClassifier(max_iter=120, max_depth=4, learning_rate=0.1, categorical_features=CATEGORICAL, random_state=0)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        self.ref = None
        self.report = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, seed=0):
        Xtr, Xrest, ytr, yrest = train_test_split(X[FEATURES], y, test_size=0.4, random_state=seed, stratify=y)
        Xcal, Xte, ycal, yte = train_test_split(Xrest, yrest, test_size=0.5, random_state=seed, stratify=yrest)
        self.gbm.fit(Xtr, ytr)
        raw_cal = self.gbm.predict_proba(Xcal)[:, 1]
        self.iso.fit(raw_cal, ycal)
        self.ref = Xtr.median()
        raw = self.gbm.predict_proba(Xte)[:, 1]
        cal = self.iso.predict(raw)
        thr = 0.5
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(cal, bins) - 1, 0, 9)
        rel = [dict(mean_pred=float(cal[idx == b].mean()), observed=float(yte.to_numpy()[idx == b].mean()), n=int((idx == b).sum()))
               for b in range(10) if (idx == b).sum() >= 5]
        pi = permutation_importance(self.gbm, Xte, yte, n_repeats=3, random_state=0, scoring="roc_auc")
        self.report = dict(
            n_train=len(Xtr), n_test=len(Xte), base_rate=float(y.mean()), auc=float(roc_auc_score(yte, cal)),
            brier_raw=float(brier_score_loss(yte, raw)), brier_calibrated=float(brier_score_loss(yte, cal)),
            precision_at_0_5=float(precision_score(yte, cal >= thr, zero_division=0)), recall_at_0_5=float(recall_score(yte, cal >= thr, zero_division=0)),
            reliability=rel,
            permutation_importance=sorted([dict(feature=f, importance=float(v)) for f, v in zip(FEATURES, pi.importances_mean)], key=lambda d: -d["importance"]))
        return self

    def score(self, X: pd.DataFrame):
        return self.iso.predict(self.gbm.predict_proba(X[FEATURES])[:, 1])

    def reasons(self, X: pd.DataFrame, k=3):
        """Top-k reasons per row via occlusion: probability drop when a feature is set to its reference value."""
        base = self.score(X)
        drops = np.zeros((len(X), len(FEATURES)))
        for j, f in enumerate(FEATURES):
            Z = X[FEATURES].copy()
            Z[f] = self.ref[f]
            drops[:, j] = base - self.score(Z)
        out = []
        for i in range(len(X)):
            order = np.argsort(-drops[i])[:k]
            out.append([describe(FEATURES[j], X.iloc[i][FEATURES[j]]) for j in order if drops[i, j] > 0])
        return out

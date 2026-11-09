"""T-learner uplift model.

Why uplift and not churn risk: the customers MOST likely to churn are often not the ones whose behaviour a credit
changes (Ascarza 2018, 'Retention Futility'); targeting by risk misallocates spend. We fit one model on treated and one on
control customers and rank by predicted RESPONSE tau(x) = P(retain | credit) - P(retain | no auto-credit).
Inputs are plain arrays (no ground truth columns); outcomes come from a randomized pilot.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = ["seg", "tenure_days", "n_orders", "avg_amount", "n_late", "n_contacts", "exposure_usd", "share_noticed"]


class TLearner:
    def __init__(self):
        kw = dict(max_iter=40, max_depth=2, learning_rate=0.05, min_samples_leaf=120, l2_regularization=10.0, random_state=0)   # heavily regularised: small pilot
        self.m1, self.m0 = HistGradientBoostingClassifier(**kw), HistGradientBoostingClassifier(**kw)

    def fit(self, X, t, y):
        t = np.asarray(t, bool)
        self.m1.fit(X[t], np.asarray(y)[t]); self.m0.fit(X[~t], np.asarray(y)[~t])
        return self

    def predict_uplift(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]

    def predict_base(self, X):
        return self.m0.predict_proba(X)[:, 1]


class LogitTLearner:
    """Low-variance alternative: regularised logistic regression per arm (better when the pilot is small)."""
    def __init__(self, C=0.3):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))
        self.m1, self.m0 = mk(), mk()

    @staticmethod
    def _x(X):
        X = np.asarray(X, float)
        seg = np.eye(3)[np.clip(X[:, 0].astype(int), 0, 2)]
        return np.hstack([X[:, 1:], seg, np.log1p(X[:, [3, 6]].clip(min=0))])

    def fit(self, X, t, y):
        t = np.asarray(t, bool); Z = self._x(X)
        self.m1.fit(Z[t], np.asarray(y)[t]); self.m0.fit(Z[~t], np.asarray(y)[~t])
        return self

    def predict_uplift(self, X):
        Z = self._x(X); return self.m1.predict_proba(Z)[:, 1] - self.m0.predict_proba(Z)[:, 1]

    def predict_base(self, X):
        return self.m0.predict_proba(self._x(X))[:, 1]


def qini_curve(y, t, score, n=20):
    """Cumulative incremental retained customers when targeting the top-k by score. Returns (fractions, gains)."""
    y, t = np.asarray(y, float), np.asarray(t, bool)
    order = np.argsort(-np.asarray(score))
    y, t = y[order], t[order]
    fr, gain = [0.0], [0.0]
    for f in np.linspace(1 / n, 1, n):
        k = int(round(f * len(y)))
        yt, yc = y[:k][t[:k]], y[:k][~t[:k]]
        nt, nc = max(1, len(yt)), max(1, len(yc))
        gain.append(float(yt.sum() - yc.sum() * nt / nc)); fr.append(float(f))
    return fr, gain


def qini_coefficient(fr, gain):
    """Area between the model curve and the random-targeting diagonal."""
    fr, gain = np.array(fr), np.array(gain)
    rand = gain[-1] * fr
    return float(np.trapezoid(gain - rand, fr))


def uplift_by_decile(y, t, score):
    y, t = np.asarray(y, float), np.asarray(t, bool)
    order = np.argsort(-np.asarray(score))
    out = []
    for d, idx in enumerate(np.array_split(order, 10)):
        yt, yc = y[idx][t[idx]], y[idx][~t[idx]]
        out.append(dict(decile=d + 1, n=int(len(idx)), uplift=float(yt.mean() - yc.mean()) if len(yt) and len(yc) else 0.0,
                        predicted=float(np.mean(np.asarray(score)[idx]))))
    return out

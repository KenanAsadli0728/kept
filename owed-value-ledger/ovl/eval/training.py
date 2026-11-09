"""Train every model on CPU (target: well under 2 minutes total). Labels/outcomes may come from `_ground_truth`
(they are labels, not features); features never do (see tests/test_leakage.py)."""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from .. import config, data as D, db, engine
from ..detector import nlp, redemption as rd, silent_failure as sf
from ..policy import settlement as pol, uplift as up
from ..policy.features import customer_features
from . import response


def _gt(eng, sql):
    return pd.read_sql(text(sql), eng)


def train_all(eng=None, now=None, verbose=False, pilot_seed=7, save=True, reuse=None):
    eng = eng or db.get_engine(); now = now or config.NOW
    T = {}
    t0 = time.time()

    def tick(name):
        T[name] = round(time.time() - t0, 2); t0_ = time.time()
        if verbose: print(f"  trained {name}: {T[name]}s cumulative")

    t = D.load_tables(eng)
    if reuse is not None:
        nl, silent, survival, reward = reuse["nlp"], reuse["silent"], reuse["survival"], reuse["reward"]
    else:
        nl = nlp.load_or_train(eng); import joblib as _j; nlp_df = pd.read_sql("SELECT text,label FROM nlp_training", eng)
        nl = nlp.train_baseline(nlp_df.text, nlp_df.label); _j.dump(nl, nlp.MODEL_PATH); tick("nlp_baseline")

        X = sf.build_features(t["orders"], t["contacts"])
        lab = _gt(eng, "SELECT ref_id, silent_failure FROM _ground_truth WHERE kind='order'").set_index("ref_id").silent_failure
        y = lab.reindex(X.index).fillna(0).astype(int)
        silent = sf.SilentFailureModel().fit(X, y); tick("silent_failure")

        hist = pd.read_sql("SELECT * FROM points_history", eng)
        feats = rd.member_features(t["customers"], t["redemptions"], hist, t["points"], now)
        survival = rd.SurvivalModel().fit(hist, feats)
        reward = rd.RewardModel(t["chart"].reward_name.unique()).fit(t["redemptions"]); tick("points_models")

    models = dict(nlp=nl, silent=silent, survival=survival, reward=reward, uplift=None)
    # ---- randomized pilot -> uplift model ----
    col = engine.collect(eng, now, models, t=t)
    sp = col["rows"]["settlement_policy"]["params"]
    per = {}
    for iou, cd, _ in col["items"]:
        if cd is None: continue
        per.setdefault(iou.customer_id, []).append(response.entry(iou.value_usd, pol.noticed_signal(iou, cd, sp), iou.unit))
    exposure = pd.DataFrame({cid: dict(exposure_usd=sum(x[0] for x in e), share_noticed=float(np.mean([x[1] for x in e]))) for cid, e in per.items()}).T
    F = customer_features(t, now, exposure)
    gt = _gt(eng, "SELECT ref_id, true_response, backfire, base_retention FROM _ground_truth WHERE kind='customer'").set_index("ref_id")
    rng = np.random.default_rng(pilot_seed)
    ids = sorted(per)
    tr = rng.random(len(ids)) < 0.5
    yv = np.array([rng.random() < response.p_retain(gt.loc[c], per[c], "treatment" if tt else "holdout") for c, tt in zip(ids, tr)]).astype(int)
    Xp = F.loc[ids, up.FEATURES].to_numpy(float)
    split = rng.random(len(ids)) < 0.6
    # pick the learner on the training half only (cross-fitted transformed-outcome score), then report on the held-out half
    ystar_all = yv * (tr.astype(float) - 0.5) / 0.25
    best, best_sc = None, -1e9
    for name, cls in (("gbm", up.TLearner), ("logit", up.LogitTLearner)):
        ia = np.where(split)[0]; half = len(ia) // 2
        m_ = cls().fit(Xp[ia[:half]], tr[ia[:half]], yv[ia[:half]])
        sc = float(np.mean(m_.predict_uplift(Xp[ia[half:]]) * ystar_all[ia[half:]]))
        if sc > best_sc: best, best_sc = (name, cls), sc
    learner_name, LearnerCls = best
    tl = LearnerCls().fit(Xp[split], tr[split], yv[split])
    s_up = tl.predict_uplift(Xp[~split]); s_risk = 1 - tl.predict_base(Xp[~split])
    fr, q_up = up.qini_curve(yv[~split], tr[~split], s_up)
    _, q_risk = up.qini_curve(yv[~split], tr[~split], s_risk)
    pilot = dict(n=len(ids), n_test=int((~split).sum()), qini=dict(fractions=fr, uplift_model=q_up, churn_risk=q_risk,
                 random=[q_up[-1] * f for f in fr]), qini_coef_uplift=up.qini_coefficient(fr, q_up), qini_coef_risk=up.qini_coefficient(fr, q_risk),
                 deciles=up.uplift_by_decile(yv[~split], tr[~split], s_up))
    dec = pilot["deciles"]
    pr_, ob_ = np.array([d["predicted"] for d in dec]), np.array([d["uplift"] for d in dec])
    ystar = yv[~split] * (tr[~split].astype(float) - 0.5) / 0.25            # transformed outcome: E[Y*|x] = tau(x) under 50/50 randomisation
    raw_slope = float((s_up * ystar).sum() / max(1e-9, (s_up ** 2).sum()))
    se = float(np.sqrt(((s_up * (ystar - raw_slope * s_up)) ** 2).sum()) / max(1e-9, (s_up ** 2).sum()))    # robust SE of the slope
    slope = float(np.clip(raw_slope - se, 0.0, 1.0))      # CONSERVATIVE: lower confidence bound, so an unstable model spends little
    pilot_extra = dict(raw_calibration_slope=raw_slope, slope_se=se)
    pilot["calibration_slope"] = slope; pilot.update(pilot_extra)
    pilot["learner"] = learner_name
    models["uplift"] = LearnerCls().fit(Xp, tr, yv); models["uplift_shrink"] = slope; tick("uplift")
    models["training_report"] = dict(pilot=pilot, seconds=T)
    if save: joblib.dump(models, engine.BUNDLE)
    return models


if __name__ == "__main__":
    t = time.time(); train_all(verbose=True); print("total", round(time.time() - t, 1), "s")

"""90-day post-settlement simulation + every reported metric -> data/metrics.json.
All outcomes use the HIDDEN response function (eval/response.py) on SYNTHETIC data. Not real-world evidence."""
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from .. import config, data as D, db, engine, llm
from ..abuse import guard
from ..contracts.late_delivery import band_credit
from ..detector import nlp, redemption as rd, silent_failure as sf
from ..policy import optimizer, settlement as pol, uplift as up
from ..policy.features import customer_features
from . import response as R

MARGIN = 0.25
CONTACT_COST = 6.0


def _gt(eng):
    g = pd.read_sql("SELECT * FROM _ground_truth", eng)
    return g[g.kind == "customer"].set_index("ref_id"), g[g.kind == "order"].set_index("ref_id"), g[g.kind == "contact"].set_index("ref_id"), g[g.kind == "points"].set_index("ref_id")


def _entries(L):
    """customer_id -> list of (value_usd, noticed) for entries that are auto-settled or claim-open (review excluded)."""
    e = defaultdict(list)
    for r in L[L.status.isin(["settled", "open"])].itertuples():
        e[r.customer_id].append(R.entry(r.value_usd, r.noticed_flag, r.currency_or_points))
    return e


def arm_outcomes(gtc, L, policy="salience", seed=11):
    rng = np.random.default_rng(seed)
    arm = L.groupby("customer_id").experiment_arm.first()
    ent = _entries(L)
    rows = []
    for cid, e in ent.items():
        g = gtc.loc[cid]
        a = arm[cid]
        pT = R.p_retain(g, e, "treatment", policy); pH = R.p_retain(g, e, "holdout", policy)
        p = pT if a == "treatment" else pH
        ret = rng.random() < p
        rows.append(dict(cid=cid, arm=a, pT=pT, pH=pH, p=p, retained=int(ret), spend=g.spend_90d * ret, spend_exp=g.spend_90d * p,
                         spendT=g.spend_90d * pT, spendH=g.spend_90d * pH, payT=R.exp_payout(e, "treatment"), payH=R.exp_payout(e, "holdout"),
                         contH=R.exp_contacts(e, "holdout"), noticed_share=np.mean([x[1] for x in e]), backfire=g.backfire))
    return pd.DataFrame(rows)


def diff_means(d, col):
    a, b = d[d.arm == "treatment"][col], d[d.arm == "holdout"][col]
    se = np.sqrt(a.var() / max(1, len(a)) + b.var() / max(1, len(b)))
    return float(a.mean() - b.mean()), float(se)


def compute_metrics(eng=None, now=None, models=None, verbose=False):
    t0 = time.time()
    eng = eng or db.get_engine(); now = now or config.NOW
    models = models or engine.load_models()
    L = pd.read_sql("SELECT * FROM ledger_entries", eng, parse_dates=["event_at", "settled_at"])
    gtc, gto, gtk, gtp = _gt(eng)
    t = D.load_tables(eng, adjusted=False)      # raw points: eligibility is evaluated BEFORE the ledger converts/extends them
    M = {"disclaimer": "ALL numbers come from SYNTHETIC data and an ASSUMED hidden response function. Real impact needs a randomized holdout on live customers.",
         "generated_for_clock": str(now)}
    trt, hol = L[L.experiment_arm == "treatment"], L[L.experiment_arm == "holdout"]

    # ---------------- coverage / time-to-settle / contacts ----------------
    done = trt[trt.status == "settled"]
    contact_orders = set(t["contacts"].order_id.dropna().astype(int))
    zero_contact = done[~done.apply(lambda r: r.contract_type in ("late_delivery", "silent_failure", "incident") and int(r.source_ref.split(":")[-1]) in contact_orders, axis=1)]
    M["coverage"] = dict(auto_settled_share=len(done) / max(1, len(trt)), zero_contact_share=len(zero_contact) / max(1, len(trt)),
                         review_share=float((trt.status == "review").mean()), n_treatment_entries=len(trt), n_holdout_entries=len(hol))
    tts = (done.settled_at - done.event_at).dt.total_seconds() / 3600
    M["time_to_settle_hours"] = dict(median=float(tts.median()), p90=float(tts.quantile(0.9)),
                                     holdout_claim_based_note="Claim-based handling depends on the customer filing; not auto-settled.")

    # ---------------- 90-day simulation ----------------
    d = arm_outcomes(gtc, L)
    dr, dr_se = diff_means(d, "retained"); ds, ds_se = diff_means(d, "spend")
    T, H = d[d.arm == "treatment"], d[d.arm == "holdout"]
    ate = float((T.pT - T.pH).mean())
    pp = float(d.retained.mean())
    n_need = 2 * (1.96 + 0.84) ** 2 * pp * (1 - pp) / max(ate, 1e-4) ** 2          # per arm, 80% power, 5% two-sided
    power = dict(minimum_detectable_uplift_pp=float(2.8 * dr_se * 100), oracle_uplift_pp=float(ate * 100), customers_needed_per_arm=int(n_need),
                 holdout_customers_now=int(len(H)),
                 eligible_customers_needed_at_10pct_holdout=int(n_need / 0.10), eligible_customers_needed_at_20pct_holdout=int(n_need / 0.20),
                 reading="a 10% holdout of this population cannot detect the effect; pool several quarters or raise the holdout share")
    ate_spend = float((T.spendT - T.spendH).mean())
    pay_extra = float((T.payT - T.payH).mean())
    saved_contacts = float(T.contH.mean())
    net_pc = ate_spend * R.VALUE_HORIZON * MARGIN - pay_extra + saved_contacts * CONTACT_COST
    M["power"] = power
    M["simulation"] = dict(
        n_treatment=len(T), n_holdout=len(H),
        retention=dict(treatment=float(T.retained.mean()), holdout=float(H.retained.mean()), observed_uplift=dr, observed_se=dr_se, oracle_uplift=ate),
        spend_90d=dict(treatment=float(T.spend.mean()), holdout=float(H.spend.mean()), observed_uplift=ds, observed_se=ds_se, oracle_uplift=ate_spend),
        payout_per_customer=dict(treatment=float(T.payT.mean()), holdout_counterfactual=float(T.payH.mean())),
        contact_deflection=dict(expected_contacts_avoided=float(T.contH.sum()), per_treated_customer=saved_contacts, saved_usd=float(T.contH.sum() * CONTACT_COST)),
        net_margin=dict(per_customer=float(net_pc), total=float(net_pc * len(T)), margin_rate=MARGIN, horizon_multiplier=R.VALUE_HORIZON, reward_cost_ratio=R.REWARD_COST_RATIO, note="oracle 90d spend uplift x 4 (12-month value) x margin - extra brand cost + contacts avoided"))

    # break-even: how many times stronger must the (assumed) response be for the ledger to pay for itself?
    def net_at(m):
        g2 = gtc.copy(); g2["true_response"] = g2.true_response * m
        dd = arm_outcomes(g2, L); T_ = dd[dd.arm == "treatment"]
        return float((T_.spendT - T_.spendH).mean() * R.VALUE_HORIZON * MARGIN - (T_.payT - T_.payH).mean() + T_.contH.mean() * CONTACT_COST)
    M["simulation"]["net_margin_vs_response_multiplier"] = [dict(multiplier=m, net_per_customer=net_at(m)) for m in (0.5, 1, 2, 3, 5, 10)]
    gain_c = ((T.set_index("cid").spendT - T.set_index("cid").spendH) * R.VALUE_HORIZON * MARGIN)
    tw = trt[trt.status.isin(["settled", "open"])].copy()
    tw["w"] = tw.value_usd / tw.groupby("customer_id").value_usd.transform("sum").clip(lower=1e-9)
    tw["benefit"] = tw.customer_id.map(gain_c).fillna(0) * tw.w                     # retention benefit shared in proportion to credited value
    bc = []
    for ct, g_ in trt.groupby("contract_type"):
        cost = g_.apply(lambda r: r.value_usd * (R.REWARD_COST_RATIO if r.currency_or_points == "POINTS" else 1.0), axis=1)
        cp = g_.noticed_flag.map(lambda n: R.claim_p(bool(n)))
        bc.append(dict(contract=ct, entries=len(g_), brand_cost=float(cost.sum()), claim_based_cost=float((cost * cp).sum()),
                       contacts_avoided=float(cp.sum()), operating_net=float(cp.sum() * CONTACT_COST - (cost.sum() - (cost * cp).sum())),
                       retention_benefit=float(tw[tw.contract_type == ct].benefit.sum())))
        bc[-1]["full_net"] = bc[-1]["operating_net"] + bc[-1]["retention_benefit"]
    M["by_contract"] = sorted(bc, key=lambda x: x["full_net"])
    M["recommended_contracts"] = [b["contract"] for b in bc if b["full_net"] > 0]

    # ---------------- breakage ----------------
    ex = L[(L.contract_type == "expiring_points")]
    expiring_all = t["points"][(t["points"].expires_at - now).dt.total_seconds().between(0, 7 * 86400) & (t["points"].balance >= 500)]
    conv = ex[ex.status == "settled"]
    will = gtp.reindex(conv.customer_id).will_redeem.fillna(0).to_numpy()
    M["breakage"] = dict(points_converted=float(conv.amount.sum()), value_usd=float(conv.value_usd.sum()), members_converted=int(len(conv)),
                         eligible_members=int(len(expiring_all)), unwanted_conversion_rate=float(will.mean()) if len(will) else 0.0,
                         note="unwanted = member would have redeemed on their own (ground truth)")

    # ---------------- abuse ----------------
    col = engine.collect(eng, now, models, t=t)
    rows = col["rows"]; ap = rows["abuse_guard"]["params"]
    claims, payout = {}, {}
    for iou, _, _ in col["items"]:
        if iou.contract_type in pol.CLAIM_TYPES and (now - iou.event_at).days <= 30:
            claims[iou.customer_id] = claims.get(iou.customer_id, 0) + 1; payout[iou.customer_id] = payout.get(iou.customer_id, 0.0) + iou.value_usd
    ab = guard.score(t["customers"], claims, payout, ap, now)
    truth = gtc.is_ring.reindex(ab.index).fillna(0).astype(bool)

    def pr(flag):
        tp = int((flag & truth).sum()); fp = int((flag & ~truth).sum()); fn = int((~flag & truth).sum())
        return dict(precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn), flagged=int(flag.sum()), tp=tp, fp=fp, fn=fn)
    rid = gtc.ring_id.reindex(ab.index)
    groups = {"coordinated_rings": rid >= 0, "lone_heavy_abusers": rid == -1, "stealth_mild_abusers": rid == -2}
    by_group = {k: dict(n=int(m.sum()), combined_recall=float(ab.flagged[m].mean()), graph_recall=float((ab.ring >= 0.5)[m].mean()),
                        anomaly_recall=float(ab.anomaly_flag[m].mean()), velocity_recall=float(ab.velocity[m].mean())) for k, m in groups.items()}
    ring_ids = set(truth[truth].index)
    nonring = L[~L.customer_id.isin(ring_ids)]
    M["abuse"] = dict(by_group=by_group, combined=pr(ab.flagged), graph_only=pr(ab.ring >= 0.5), anomaly_only=pr(ab.anomaly_flag), velocity_only=pr(ab.velocity),
                      false_positive_payout_rate=float(nonring[nonring.status == "review"].value_usd.sum() / max(1e-9, nonring.value_usd.sum())),
                      ring_value_auto_settled_usd=float(L[L.customer_id.isin(ring_ids) & (L.status == "settled")].value_usd.sum()),
                      ring_value_held_usd=float(L[L.customer_id.isin(ring_ids) & (L.status == "review")].value_usd.sum()),
                      weights=dict(ring=guard.W_RING, anomaly=guard.W_ANOM, flag_at=guard.FLAG_AT),
                      denials_automatic=int((L.status == "denied").sum()))

    # ---------------- salience guardrail ----------------
    scen = {}
    for pol_name in ("salience", "notify_all"):
        dd = arm_outcomes(gtc, L, policy=pol_name)
        T2 = dd[dd.arm == "treatment"]
        scen[pol_name] = dict(oracle_retention_uplift=float((T2.pT - T2.pH).mean()),
                              net_margin_per_customer=float(((T2.spendT - T2.spendH).mean()) * R.VALUE_HORIZON * MARGIN - (T2.payT - T2.payH).mean() + T2.contH.mean() * CONTACT_COST))
        u = T2[T2.noticed_share == 0]
        scen[pol_name]["silent_only_customers_uplift"] = float((u.pT - u.pH).mean()) if len(u) else None
        scen[pol_name]["backfire_group_uplift"] = float((T2[T2.backfire == 1].pT - T2[T2.backfire == 1].pH).mean())
    ex_u = d[(d.arm == "treatment") & (d.noticed_share == 1)]
    sl_u = d[(d.arm == "treatment") & (d.noticed_share == 0)]
    M["salience_guardrail"] = dict(policies=scen, explicit_only_customers_uplift=float((ex_u.pT - ex_u.pH).mean()) if len(ex_u) else None,
                                   silent_only_customers_uplift=float((sl_u.pT - sl_u.pH).mean()) if len(sl_u) else None,
                                   n_explicit_only=len(ex_u), n_silent_only=len(sl_u),
                                   reading="notify_all = apologise/alert for failures the customer never noticed; it erodes the gain (HBR 2026, Ascarza 2016).")

    # ---------------- sensitivity: late-delivery payout cap ----------------
    late = trt[(trt.contract_type == "late_delivery") & (trt.status == "settled")].copy()
    late["order_id"] = late.source_ref.str.split(":").str[-1].astype(int)
    od = t["orders"].set_index("id")
    bands = rows["late_delivery"]["params"]["bands"]
    delay = ((od.loc[late.order_id].delivered_at.values - od.loc[late.order_id].promised_at.values) / np.timedelta64(1, "m")).astype(float)
    amt = od.loc[late.order_id].amount.values
    sens = []
    ent_base = _entries(L)
    for cap in [1, 2, 3, 5, 8, 12]:
        newv = [min(cap, band_credit(dl, am, bands)[1]) if band_credit(dl, am, bands)[2] is not None else 0 for dl, am in zip(delay, amt)]
        delta = dict(zip(late.id, np.array(newv) - late.value_usd.values))
        lv = late.assign(v=np.array(newv)).groupby("customer_id").v.sum()
        gains, costs = 0.0, 0.0
        for cid in T.cid:
            e = ent_base[cid]
            e2 = list(e)
            if cid in lv.index:
                old = late[late.customer_id == cid].value_usd.sum()
                shift = lv[cid] - old
                big = max(x[0] for x in e)
                e2 = [(max(0.0, v + shift * (v == big)), n, max(0.0, c + shift * (v == big))) for v, n, c in e]   # shift applied to the largest entry (approximation)
            g_ = gtc.loc[cid]
            pT_ = R.p_retain(g_, e2, "treatment"); pH_ = R.p_retain(g_, e, "holdout")
            gains += (pT_ - pH_) * g_.spend_90d * R.VALUE_HORIZON * MARGIN
            costs += R.exp_payout(e2, "treatment") - R.exp_payout(e, "holdout")
        sens.append(dict(cap=cap, net_margin_per_customer=float((gains - costs + T.contH.sum() * CONTACT_COST) / max(1, len(T)))))
    M["sensitivity_late_cap"] = sens

    # ---------------- settlement optimizer: budget vs net margin + fairness ----------------
    sp = rows["settlement_policy"]["params"]
    base_ent = L[(L.experiment_arm == "treatment") & (L.status == "settled") & (L.currency_or_points == "USD")].copy()
    base_ent["base"] = base_ent.value_usd - base_ent.topup.fillna(0)
    owed = base_ent.groupby("customer_id").base.sum()
    exp_df = L.assign(n=L.noticed_flag).groupby("customer_id").agg(exposure_usd=("value_usd", "sum"), share_noticed=("n", "mean"))
    F = customer_features(t, now, exp_df)
    tau = pd.Series(models["uplift"].predict_uplift(F.loc[owed.index, up.FEATURES]) * models.get("uplift_shrink", 1.0), index=owed.index)
    odf = pd.DataFrame({"owed_usd": owed, "uplift_hat": tau, "spend_hat": F.loc[owed.index, "spend_hat"]})
    curve = []
    raw_tau = pd.Series(models["uplift"].predict_uplift(F.loc[owed.index, up.FEATURES]), index=owed.index)
    shrink = models.get("uplift_shrink", 1.0)
    oracle_tau = gtc.true_response.reindex(owed.index) * R.g(owed)       # tau_hat equivalent if the true response were known

    def true_gain(top):
        tot = 0.0
        for cid, tv in top[top > 0].items():
            g_ = gtc.loc[cid]
            tot += MARGIN * R.VALUE_HORIZON * g_.spend_90d * g_.true_response * (R.g(owed[cid] + tv) - R.g(owed[cid])) - tv
        return float(tot)
    for B in [0, 10, 25, 50, 75, 100, 150, 250]:
        row = dict(budget=B)
        for name, tv in (("model_raw", raw_tau), ("model_calibrated", raw_tau * shrink), ("oracle", oracle_tau)):
            top = optimizer.allocate(odf.assign(uplift_hat=tv), B, sp["topup_pct"], sp["topup_cap"])
            row[name + "_spent"] = float(top.sum()); row[name + "_true_net_gain"] = true_gain(top)
            if name == "model_raw": row["model_raw_predicted_net_gain"] = float(top.attrs["predicted_gain"] - top.sum())
        full = np.minimum(sp["topup_cap"], sp["topup_pct"] * odf.owed_usd)
        flat = full * min(1.0, B / max(1e-9, full.sum()))
        row["untargeted_true_net_gain"] = true_gain(flat)
        curve.append(row)
    top_cfg = optimizer.allocate(odf.assign(uplift_hat=raw_tau * shrink), sp["budget_usd"], sp["topup_pct"], sp["topup_cap"])
    cu = t["customers"].set_index("id")
    groups = pd.DataFrame({"segment": cu.segment.reindex(odf.index),
                           "shared_address": (cu.groupby("address_hash").segment.transform("size") > 1).reindex(odf.index).map({True: "yes", False: "no"})})
    M["optimizer"] = dict(budget_curve=curve, configured_budget=sp["budget_usd"], spent_at_configured_budget=float(top_cfg.sum()),
                          fairness=optimizer.fairness_audit(odf.assign(uplift_hat=raw_tau), optimizer.allocate(odf.assign(uplift_hat=raw_tau), sp["budget_usd"], sp["topup_pct"], sp["topup_cap"]), groups, sensitive=["shared_address"]),
                          calibration_slope=shrink,
                          note="Top-ups only; the contract amount is always paid. Dose-response g(c)=1-exp(-c/4) is an ASSUMPTION the live holdout must re-estimate.")

    # ---------------- AI components ----------------
    M["models"] = model_eval(eng, t, models, gtk, gto, gtp, col, ab, truth, L)
    M["uplift"] = models["training_report"]["pilot"]
    from . import training
    rob = []
    for sd in range(1, 9):                      # repeat the randomized pilot: is the uplift model stable?
        pp_ = training.train_all(eng, now, pilot_seed=sd, save=False, reuse=models)["training_report"]["pilot"]
        rob.append(dict(seed=sd, learner=pp_["learner"], qini_uplift=pp_["qini_coef_uplift"], qini_churn_risk=pp_["qini_coef_risk"], calibrated_slope=pp_["calibration_slope"]))
    M["uplift_robustness"] = dict(runs=rob, uplift_beats_risk=int(sum(r["qini_uplift"] > r["qini_churn_risk"] for r in rob)), of=len(rob),
                                  median_slope=float(np.median([r["calibrated_slope"] for r in rob])),
                                  reading="the model is unstable across pilot draws; the conservative slope shrinks its top-ups accordingly")
    M["notes"] = note_stats(L)
    M["runtime_seconds"] = round(time.time() - t0, 1)
    config.METRICS_PATH.write_text(json.dumps(M, indent=1, default=float), encoding="utf8")
    return M


def note_stats(L):
    c = L.note_source.value_counts().to_dict()
    att = c.get("llm", 0) + c.get("template_after_reject", 0)
    return dict(sources=c, llm_attempts=att, rejection_rate=(c.get("template_after_reject", 0) / att) if att else None,
                status="LLM disabled (offline): every note uses the deterministic template" if not att else "LLM-phrased notes with guardrail")


def model_eval(eng, t, models, gtk, gto, gtp, col, ab, truth, L):
    out = {}
    # --- NLP: LLM vs baseline ---
    c = t["contacts"].set_index("id")
    labels = gtk.label.reindex(c.index)
    c = c[labels.notna()]                     # live-demo contacts have no ground-truth label
    labels = labels[labels.notna()]
    pred = models["nlp"].predict(c.text.tolist())
    inf = t["contacts"].set_index("id")
    inf = inf[inf.link_inferred]
    truth_link = gtk.true_order_id.reindex(inf.index)
    nlp_res = dict(order_link_inference=dict(n_inferred=int(len(inf)), accuracy=float((inf.order_id == truth_link).mean())), n=len(c), baseline_macro_f1=float(f1_score(labels, pred, average="macro")), baseline_accuracy=float((labels.values == pred).mean()))
    if llm.available():
        smp = c.sample(min(150, len(c)), random_state=0)
        outs = [nlp.extract(x, models["nlp"]) for x in smp.text]
        used = [s for _, s in outs]
        nlp_res["llm"] = dict(n=len(smp), valid_json_rate=float(np.mean([s == "llm" for s in used])),
                              macro_f1=float(f1_score(labels.reindex(smp.index), [e.failure_type for e, _ in outs], average="macro")))
    else:
        nlp_res["llm"] = dict(available=False, reason="KEPT_LLM=1 and ANTHROPIC_API_KEY not set - the TF-IDF baseline is the live path")
    out["nlp"] = nlp_res
    # --- incidents ---
    truth_v = gto[gto.incident_id.notna()]
    res = []
    for inc in col["incidents"]:
        found = set(inc["victim_orders"])
        best = max(truth_v.incident_id.unique(), key=lambda k: len(found & set(truth_v[truth_v.incident_id == k].index)))
        gtset = set(truth_v[truth_v.incident_id == best].index)
        res.append(dict(id=inc["id"], matches=best, affected_customers=inc["n_affected"], complained=inc["n_complained"], silent_victims=inc["n_silent"],
                        failure_type=inc["failure_type"], order_precision=len(found & gtset) / max(1, len(found)), order_recall=len(found & gtset) / max(1, len(gtset)),
                        embed_backend=inc["embed_backend"], silent_credited=int(((L.contract_type == "incident") & L.source_ref.str.startswith(inc["id"])).sum())))
    out["incidents"] = dict(discovered=res, planted=int(truth_v.incident_id.nunique()))
    # --- silent failure ---
    out["silent_failure"] = models["silent"].report
    # --- points ---
    hist = pd.read_sql("SELECT * FROM points_history", eng)
    feats = rd.member_features(t["customers"], t["redemptions"], hist, t["points"], config.NOW)
    soon = feats[(feats.days_left >= 0) & (feats.days_left <= 7) & (feats.balance >= 500)]
    pn = models["survival"].p_not_redeem(soon)
    will = gtp.will_redeem.reindex(soon.index).fillna(0).to_numpy()
    thr = col["rows"]["expiring_points"]["params"]["min_p_not_redeem"]
    sel = pn >= thr
    out["points"] = dict(survival=models["survival"].report, reward=models["reward"].report, n_eligible=len(soon),
                         auc_vs_truth=float(roc_auc_score(1 - will, pn)) if len(set(will)) > 1 else None, threshold=thr,
                         convert_all_unwanted_rate=float(will.mean()), thresholded_unwanted_rate=float(will[sel].mean()) if sel.any() else 0.0,
                         conversions_kept=int(sel.sum()), conversions_avoided=int((~sel).sum()))
    return out

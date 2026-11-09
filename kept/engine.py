"""The nightly pipeline: contracts + detectors -> policy -> abuse guard -> optimizer -> notes -> ledger (idempotent)."""
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from . import config, data as D, db
from .abuse import guard
from .contracts import evaluate_all
from .contracts.base import IOU, money
from .detector import incidents as inc_mod, nlp, redemption as rd, silent_failure as sf
from .policy import notes, optimizer, settlement as pol
from .policy.features import customer_features

BUNDLE = config.MODELS_DIR / "bundle.joblib"


def load_models():
    return joblib.load(BUNDLE) if BUNDLE.exists() else None


# --------------------------------------------------------------------------------------------
def collect(eng, now, models, t=None, rows=None, only=None):
    """Detect every IOU (no policy yet). Returns dict(items=[(iou, ctx, detector_reasons)], incidents=[...])."""
    t = t or D.load_tables(eng)
    rows = rows or db.load_contract_rows(eng)
    hist = pd.read_sql("SELECT * FROM points_history", eng)
    feats = rd.member_features(t["customers"], t["redemptions"], hist, t["points"], now)
    soon = feats[(feats.days_left >= 0) & (feats.days_left <= 7)]
    pnr = dict(zip(soon.index, models["survival"].p_not_redeem(soon))) if len(soon) else {}
    names = t["redemptions"].sort_values("redeemed_at").groupby("customer_id").reward_name.apply(list).to_dict()

    items, ctx = [], {}
    for cd in D.iter_customer_data(t, now, only):
        cid = cd["customer"]["id"]
        cd["p_not_redeem"] = pnr.get(cid)
        cd["predicted_reward"] = models["reward"].predict(names.get(cid, []))
        ctx[cid] = cd
        for iou in evaluate_all(cd, rows):
            items.append((iou, cd, None))

    late_orders = {i.meta["order_id"] for i, _, _ in items if i.contract_type == "late_delivery"}
    # ---- silent-failure detector ----
    sfp = rows["silent_failure"]
    if sfp["active"]:
        P = sfp["params"]
        X = sf.build_features(t["orders"], t["contacts"])
        od = t["orders"].set_index("id")
        recent = od.loc[X.index].delivered_at >= now - pd.Timedelta(days=P["lookback_days"])
        X = X[recent.values]
        if only is not None:
            X = X[X.customer_id.isin(only)]
        if len(X):
            p = models["silent"].score(X)
            sel = X[p >= P["threshold"]]
            rs = models["silent"].reasons(sel, k=3) if len(sel) else []
            for (oid, r), why, pp in zip(sel.iterrows(), rs, p[p >= P["threshold"]]):
                o = od.loc[oid]
                amt = money(min(P["credit_cap"], P["credit_pct"] * o.amount))
                if amt <= 0 or int(oid) in late_orders: continue          # already compensated as late: do not double-pay the same order
                iou = IOU("silent_failure", f"order:{oid}", amt, "USD", amt, o.delivered_at.to_pydatetime(),
                          f"Order #{oid} was flagged as likely damaged or faulty. ${amt:.2f} credited.",
                          dict(order_id=int(oid), probability=float(pp)), customer_id=int(r.customer_id))
                items.append((iou, ctx.get(int(r.customer_id)), why))

    # ---- incident discovery (embedding clustering) + silent-victim propagation ----
    incidents = []
    ip = rows["incident"]
    if ip["active"]:
        P = ip["params"]
        nl = models["nlp"]
        incidents = inc_mod.discover(t["contacts"], t["orders"], nl, P, now)
        od = t["orders"].set_index("id")
        for inc in incidents:
            comp = set(inc["complaint_customers"])
            for oid in inc["victim_orders"]:
                o = od.loc[oid]; cid = int(o.customer_id)
                if only is not None and cid not in only: continue
                amt = money(min(P["credit_cap"], P["credit_pct"] * o.amount))
                ev = (o.delivered_at if pd.notna(o.delivered_at) else o.placed_at).to_pydatetime()
                items.append((IOU("incident", f"{inc['id']}:{oid}", amt, "USD", amt, ev,
                                  f"Order #{oid} was part of operations incident {inc['id']}. ${amt:.2f} credited.",
                                  dict(order_id=int(oid), incident_id=inc["id"], complained=cid in comp), customer_id=cid), ctx.get(cid), None))
    return dict(items=items, incidents=incidents, tables=t, ctx=ctx, rows=rows)


def claims_from_ledger(eng, now):
    q = text("SELECT customer_id, COUNT(*), SUM(value_usd) FROM ledger_entries WHERE contract_type IN ('late_delivery','silent_failure','incident') AND event_at >= :t GROUP BY customer_id")
    with eng.connect() as c:
        rows = c.execute(q, dict(t=(now - pd.Timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"))).fetchall()
    return {r[0]: r[1] for r in rows}, {r[0]: float(r[2]) for r in rows}


# --------------------------------------------------------------------------------------------
def run_pipeline(eng=None, now=None, only=None, models=None, realtime=False):
    eng = eng or db.get_engine()
    now = now or config.NOW
    models = models or load_models()
    if models is None:
        from .eval import training
        models = training.train_all(eng, now)
    col = collect(eng, now, models, only=only)
    items, t, rows = col["items"], col["tables"], col["rows"]
    sp, ap = rows["settlement_policy"]["params"], rows["abuse_guard"]["params"]

    with eng.connect() as c:
        existing = {(r[0], r[1]) for r in c.execute(text("SELECT contract_type, source_ref FROM ledger_entries"))}
    items = [x for x in items if (x[0].contract_type, x[0].source_ref) not in existing]

    # ---- policy: noticed / mode / arm / reasons ----
    recs = []
    for iou, cd, why in items:
        if cd is None: continue
        noticed = pol.noticed_signal(iou, cd, sp)
        exp = bool(iou.meta.get("experimental"))
        mode = pol.decide_mode(noticed, exp)
        recs.append(dict(iou=iou, cd=cd, noticed=noticed, mode=mode, arm=pol.assign_arm(iou.customer_id, sp["holdout_pct"]),
                         reasons=pol.reasons_for(iou, cd, noticed, why)))

    # ---- abuse guard (graph + anomaly + velocity) on the customers in scope ----
    cust = t["customers"]
    claims, payout = {}, {}
    if only is None:                                # full run: the detection set IS the complete picture
        for iou, _, _ in col["items"]:
            if iou.contract_type in pol.CLAIM_TYPES and (now - iou.event_at).days <= 30:
                claims[iou.customer_id] = claims.get(iou.customer_id, 0) + 1
                payout[iou.customer_id] = payout.get(iou.customer_id, 0.0) + iou.value_usd
    else:                                           # partial run: history from the ledger + the new items
        claims, payout = claims_from_ledger(eng, now)
        for iou, _, _ in items:
            if iou.contract_type in pol.CLAIM_TYPES and (now - iou.event_at).days <= 30:
                claims[iou.customer_id] = claims.get(iou.customer_id, 0) + 1
                payout[iou.customer_id] = payout.get(iou.customer_id, 0.0) + iou.value_usd
    ab = guard.score(cust, claims, payout, ap, now)

    # ---- uplift-sized top-ups via the constrained optimizer (treatment arm, cash entries) ----
    topups = {}
    uplift = models.get("uplift")
    elig = [r for r in recs if r["arm"] == "treatment" and r["iou"].unit == "USD" and not ab.loc[r["iou"].customer_id, "flagged"]
            and r["iou"].value_usd <= ap["review_payout_usd"]]
    if uplift is not None and elig and sp["topup_pct"] > 0:
        exp_df = pd.DataFrame([dict(customer_id=r["iou"].customer_id, v=r["iou"].value_usd, n=int(r["noticed"])) for r in recs]).groupby("customer_id") \
            .agg(exposure_usd=("v", "sum"), share_noticed=("n", "mean"))
        F = customer_features(t, now, exp_df)
        cids = sorted({r["iou"].customer_id for r in elig})
        from .policy.uplift import FEATURES as UF
        tau = pd.Series(uplift.predict_uplift(F.loc[cids, UF]) * models.get("uplift_shrink", 1.0), index=cids)   # shrunk by pilot calibration slope
        owed = pd.Series({cid: sum(r["iou"].value_usd for r in elig if r["iou"].customer_id == cid) for cid in cids})
        odf = pd.DataFrame({"owed_usd": owed, "uplift_hat": tau, "spend_hat": F.loc[cids, "spend_hat"]})
        top = optimizer.allocate(odf, sp["budget_usd"], sp["topup_pct"], sp["topup_cap"])
        for cid, v in top.items():
            if v > 0: topups[cid] = (float(v), float(tau[cid]))

    # ---- write ledger ----
    inserted = 0
    flags_in_scope = []
    with eng.begin() as c:
        for r in recs:
            iou, cid = r["iou"], r["iou"].customer_id
            amount, value, topup, tau_hat = iou.amount, iou.value_usd, 0.0, None
            if cid in topups and iou.unit == "USD" and r["arm"] == "treatment":
                best = max((x for x in recs if x["iou"].customer_id == cid and x["iou"].unit == "USD" and x["arm"] == "treatment"), key=lambda x: x["iou"].value_usd)
                if best is r:
                    topup, tau_hat = topups[cid]
                    amount = money(amount + topup); value = money(value + topup)
            tmpl = iou.note + (f" Includes an additional ${topup:.2f} credit." if topup else "")
            amt_str = f"${amount:.2f}" if iou.unit == "USD" else f"{int(amount):,}"
            note, nsrc = notes.make_note(tmpl, amt_str)
            a = ab.loc[cid]
            status, review_reason = "settled", None
            if a.flagged:
                status, review_reason = "review", (a.reason or "abuse_score")
            elif value > ap["review_payout_usd"] and iou.contract_type != "expiring_points":
                status, review_reason = "review", "large_payout"
            elif r["arm"] == "holdout":
                status = "open"                                   # claim-based handling: nothing is auto-settled
            settled_at = None
            if status == "settled":
                settled_at = now if realtime else min(now, pol.next_nightly(iou.event_at))
            res = c.execute(text("""INSERT OR IGNORE INTO ledger_entries(customer_id,contract_type,source_ref,amount,currency_or_points,value_usd,status,
                noticed_flag,settlement_mode,note,reasons,note_source,event_at,created_at,settled_at,experiment_arm,review_reason,uplift_score,topup)
                VALUES(:cid,:ct,:sr,:am,:u,:v,:st,:nt,:md,:note,:rs,:ns,:ev,:cr,:sa,:arm,:rr,:up,:tu)"""),
                dict(cid=cid, ct=iou.contract_type, sr=iou.source_ref, am=amount, u=iou.unit, v=value, st=status, nt=int(r["noticed"]), md=r["mode"],
                     note=note, rs=json.dumps(r["reasons"]), ns=nsrc, ev=iou.event_at.strftime("%Y-%m-%d %H:%M:%S"),
                     cr=iou.event_at.strftime("%Y-%m-%d %H:%M:%S"), sa=settled_at.strftime("%Y-%m-%d %H:%M:%S") if settled_at else None,
                     arm=r["arm"], rr=review_reason, up=tau_hat, tu=topup))
            inserted += res.rowcount
            if res.rowcount and status == "settled" and iou.contract_type == "expiring_points":
                c.execute(text("INSERT OR IGNORE INTO points_adjustments(customer_id,used_points,extend_days,ledger_ref) VALUES(:c,:u,:e,:r)"),
                          dict(c=cid, u=int(iou.meta["points_used"]), e=int(iou.meta.get("extend_days", 0)), r=iou.source_ref))
        # abuse flags for customers in scope
        scope = list(only) if only is not None else list(ab.index)
        c.execute(text(f"DELETE FROM abuse_flags WHERE customer_id IN ({','.join(str(int(x)) for x in scope)})")) if scope else None
        for cid, a in ab.loc[ab.index.intersection(scope)].iterrows():
            if a.flagged:
                c.execute(text("INSERT INTO abuse_flags(customer_id,reason,score,created_at) VALUES(:c,:r,:s,:t)"),
                          dict(c=int(cid), r=a.reason or "abuse_score", s=float(a.abuse_score), t=now.strftime("%Y-%m-%d %H:%M:%S")))
    return dict(detected=len(col["items"]), new=len(recs), inserted=inserted, incidents=col["incidents"], abuse=ab)

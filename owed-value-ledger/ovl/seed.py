"""Deterministic synthetic data generator (5,000 customers, 12 months).

Ground truth lives ONLY in table `_ground_truth` and is used for evaluation, label generation
and the hidden response function of the simulation. It is never used as a model feature.
"""
import bisect
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import text

from . import config, db
from .detector.texts import make_text, TYPES

NOW = config.NOW
CATS = ["groceries", "electronics", "apparel", "home", "beauty", "glassware"]
CAT_P = [0.40, 0.10, 0.15, 0.15, 0.10, 0.10]
CAT_PRICE = {"groceries": (3, 14), "electronics": (60, 250), "apparel": (15, 70), "home": (12, 80), "beauty": (8, 40), "glassware": (14, 60)}
CAT_FAIL = {"groceries": 0.05, "electronics": 0.08, "apparel": 0.02, "home": 0.03, "beauty": 0.02, "glassware": 0.09}
CARRIERS = ["A", "B", "C"]
WAREHOUSES = ["W1", "W2", "W3", "W4"]
REWARDS = [  # name, cost_v1, cost_v2, cash value
    ("Iced latte", 150, 200, 4.5), ("Pastry of the day", 120, 160, 3.6), ("Free delivery pass", 200, 200, 6.0),
    ("$5 basket credit", 250, 250, 5.0), ("Free dessert", 180, 180, 5.4), ("Meal box", 600, 600, 18.0)]
SUBS = ["Streaming add-on", "Meal kit", "Premium delivery", "Music add-on"]
CORE = {"late_delivery", "price_protection", "expiring_points", "unused_subscription", "devaluation_true_up"}
WEDGE = ["late_delivery", "devaluation_true_up"]      # the two promises that pay for themselves in the simulation
SEG_BASE = {"new": 0.62, "regular": 0.78, "vip": 0.88}
SEG_TAU = {"new": 0.10, "regular": 0.06, "vip": 0.02}
SEG_ORDERS = {"new": 3, "regular": 6, "vip": 12}


def iso(d):
    return None if d is None else d.strftime("%Y-%m-%d %H:%M:%S")


def build(n=None, seed=None, path=None, verbose=False):
    n = n or config.N_CUSTOMERS
    rng = np.random.default_rng(seed or config.SEED)
    eng = db.get_engine(path) if path else db.get_engine()
    T0 = NOW - timedelta(days=400)

    # ---------------- SKU catalog + price timelines ----------------
    skus, sku_cat, series = [], {}, {}
    ev_rows = []
    for cat in CATS:
        lo, hi = CAT_PRICE[cat]
        for k in range(60):
            s = f"{cat[:3].upper()}-{k:03d}"
            base = round(float(rng.uniform(lo, hi)), 2)
            skus.append(s); sku_cat[s] = cat
            times, prices = [T0], [base]
            n_drops = rng.poisson(1.3)
            for d in sorted(rng.uniform(0, 395, n_drops)):
                t = T0 + timedelta(days=float(d))
                if t <= times[-1]: continue
                pr = round(base * float(rng.uniform(0.75, 0.95)), 2)
                times.append(t); prices.append(pr)
                back = t + timedelta(days=float(rng.uniform(20, 50)))
                times.append(back); prices.append(base)
            keep = [(t, p) for t, p in zip(times, prices) if t <= NOW]
            series[s] = ([t for t, _ in keep], [p for _, p in keep])
            for t, p in keep:
                ev_rows.append((s, p, iso(t)))
    # the hero SKU
    series["ELEC-HEADPHONES"] = ([T0, NOW - timedelta(days=1)], [129.0, 109.0])
    sku_cat["ELEC-HEADPHONES"] = "electronics"
    ev_rows += [("ELEC-HEADPHONES", 129.0, iso(T0)), ("ELEC-HEADPHONES", 109.0, iso(NOW - timedelta(days=1)))]
    cat_skus = {c: [s for s in skus if sku_cat[s] == c] for c in CATS}

    def price_at(s, t):
        ts, ps = series[s]
        return ps[max(0, bisect.bisect_right(ts, t) - 1)]

    # ---------------- customers ----------------
    seg = rng.choice(["new", "regular", "vip"], n, p=[0.35, 0.50, 0.15])
    cust, gt_c = [], []
    for i in range(n):
        cid = i + 1
        s = seg[i]
        age = {"new": rng.integers(20, 150), "regular": rng.integers(150, 900), "vip": rng.integers(300, 1500)}[s]
        addr = f"addr_{cid}"
        if cid > 5 and rng.random() < 0.06:
            addr = f"addr_{int(rng.integers(2, cid))}"  # households
        cust.append(dict(id=cid, name=f"Customer {cid}", segment=s, created_at=iso(NOW - timedelta(days=int(age))),
                         device_id=f"dev_{cid}", address_hash=addr, payment_hash=f"pay_{cid}"))
    cust[0].update(name="Maya R.", segment="regular", created_at=iso(NOW - timedelta(days=400)),
                   device_id="dev_maya", address_hash="addr_maya", payment_hash="pay_maya")

    ring_of = {}
    pool = list(rng.permutation(np.arange(6, n + 1)))
    for g in range(5):
        size = int(rng.integers(6, 11))
        members = [int(pool.pop()) for _ in range(size)]
        for m, cid in enumerate(members):
            ring_of[cid] = g
            c = cust[cid - 1]
            c["device_id"] = f"ringdev_{g}_{m % 2}"
            c["address_hash"] = f"ringaddr_{g}_{(m // 2) % 2}"
            c["payment_hash"] = f"ringpay_{g}_{m % 3 == 0}"
    for cid in ring_of: cust[cid - 1]["segment"] = "new"
    lone = [int(pool.pop()) for _ in range(10)]          # 'lone' abusers: heavy claims, NO shared device/address/payment
    stealth = [int(pool.pop()) for _ in range(12)]       # stealth abusers: mild claim rates, nothing shared
    for cid in stealth: ring_of[cid] = -2; cust[cid - 1]["segment"] = "new"
    for cid in lone: ring_of[cid] = -1; cust[cid - 1]["segment"] = "new"

    # ---------------- orders / items / contacts ----------------
    orders, items, contacts, gt_o = [], [], [], []
    oid = 0

    def add_order(cid, placed, cat, ring=False, force=None):
        """force: dict overrides (delay, silent, incident, warehouse, carrier, amount, items)."""
        nonlocal oid
        force = force or {}
        oid += 1
        wh = force.get("warehouse", WAREHOUSES[int(rng.integers(4))])
        car = force.get("carrier", CARRIERS[int(rng.integers(3))])
        promised = placed + (timedelta(hours=float(rng.integers(3, 7))) if cat == "groceries" else timedelta(days=int(rng.integers(1, 4))))
        late_p = 0.35 if ring else 0.08
        if "delay" in force: delay = force["delay"]
        elif rng.random() < late_p: delay = float(np.clip(rng.lognormal(np.log(70), 0.8), 31, 500))
        elif rng.random() < 0.06: delay = float(rng.uniform(12, 30))
        else: delay = float(rng.uniform(-60, 10))
        delivered = promised + timedelta(minutes=delay)
        silent = False
        if "silent" in force: silent = force["silent"]
        elif not ring:
            p = CAT_FAIL[cat] * (1.8 if car == "C" else 1.0)
            silent = rng.random() < p
        else:
            silent = rng.random() < 0.25
        status = "delivered"
        if delivered > NOW: status, delivered = "shipped", None
        else:
            if silent: status = "returned" if rng.random() < 0.55 else "delivered"
            elif rng.random() < (0.12 if ring else 0.05): status = "returned"
        k = int(force.get("n_items", rng.integers(1, 4)))
        its = force.get("items") or [(s, price_at(s, placed)) for s in rng.choice(cat_skus[cat], k, replace=False)]
        amount = round(sum(p for _, p in its), 2)
        tv = int(rng.poisson(3.5 if delay > 30 else 0.8))
        orders.append(dict(id=oid, customer_id=cid, placed_at=iso(placed), promised_at=iso(promised), delivered_at=iso(delivered),
                           amount=amount, category=cat, status=status, carrier=car, warehouse=wh, tracking_views=force.get("tracking_views", tv)))
        for s, p in its: items.append(dict(order_id=oid, sku=s, price_paid=p))
        gt_o.append(dict(kind="order", ref_id=oid, silent_failure=int(silent and not force.get("incident")), incident_id=force.get("incident"), label=None))
        # contacts
        ref = delivered or placed
        def contact(kind, p, dly=(1, 70), label=None):
            if rng.random() < p:
                at = ref + timedelta(hours=float(rng.uniform(*dly)))
                if at < NOW:
                    contacts.append(dict(customer_id=cid, created_at=iso(at), channel=str(rng.choice(["chat", "email", "app"])),
                                         text=make_text(rng, kind, wh, car, placed.strftime("%b %d")), incident_id=None, order_id=oid))
                    gt_o.append(dict(kind="contact", ref_id=len(contacts), silent_failure=None, incident_id=force.get("incident"), label=kind, true_order_id=oid))
        if force.get("incident"):
            contact(force["complaint_kind"], force["complaint_p"])
        else:
            if silent: contact("damaged", 0.30)
            elif rng.random() < 0.03: contact("other", 1.0)
            if delay > 30: contact("late", 0.30 if delay > 120 else 0.10)
            if status == "returned" and not silent: contact("other", 0.1)
        return oid

    # incident days
    inc_specs = [dict(id="INC-W3-SPOILED", warehouse="W3", carrier=None, day=12, extra=60, kind="spoiled", p=0.30),
                 dict(id="INC-CARRIER-C-MISSCAN", warehouse=None, carrier="C", day=25, extra=150, kind="not_received", p=0.12)]

    for c in cust:
        cid = c["id"]
        if cid == config.MAYA_ID: continue
        ring = cid in ring_of
        lam = 9 if ring_of.get(cid) == -2 else (22 if ring else SEG_ORDERS[c["segment"]])
        k = int(rng.poisson(lam))
        created = pd.Timestamp(c["created_at"]).to_pydatetime()
        for _ in range(k):
            if ring: days = float(rng.uniform(0.2, 60))
            else: days = float(rng.uniform(0.2, 365))
            placed = NOW - timedelta(days=days)
            if placed < created: placed = created + timedelta(days=float(rng.uniform(0, 5)))
            if placed > NOW - timedelta(hours=1): continue
            cat = str(rng.choice(CATS, p=CAT_P))
            add_order(cid, placed, cat, ring)

    for spec in inc_specs:
        day = (NOW - timedelta(days=spec["day"])).replace(hour=0, minute=0, second=0)
        for _ in range(spec["extra"]):
            cid = int(rng.integers(6, n + 1))
            placed = day + timedelta(hours=float(rng.uniform(8, 20)))
            cat = "groceries" if spec["warehouse"] else str(rng.choice(CATS, p=CAT_P))
            force = dict(delay=float(rng.uniform(-40, 5)), silent=False, incident=spec["id"], complaint_kind=spec["kind"], complaint_p=spec["p"])
            if spec["warehouse"]: force["warehouse"] = spec["warehouse"]
            if spec["carrier"]: force["carrier"] = spec["carrier"]
            add_order(cid, placed, cat, False, force)

    # ---------------- Maya (hero persona) ----------------
    mid = config.MAYA_ID
    placed = NOW - timedelta(days=1, hours=3)   # groceries, promised 11:00 yesterday-ish, 47 min late
    o1 = add_order(mid, placed, "groceries", False, dict(delay=47, silent=False, items=[("GRO-001", 31.2), ("GRO-002", 31.2)], tracking_views=2,
                                                          warehouse="W1", carrier="A"))
    o1_row = orders[-1]
    o1_row["promised_at"] = iso(NOW - timedelta(days=1) - timedelta(hours=1)); o1_row["delivered_at"] = iso(NOW - timedelta(days=1) - timedelta(hours=1) + timedelta(minutes=47))
    contacts.append(dict(customer_id=mid, created_at=iso(NOW - timedelta(days=1) - timedelta(minutes=30)), channel="chat",
                         text="Order still hasn't arrived and it was due an hour ago", incident_id=None, order_id=o1))
    gt_o.append(dict(kind="contact", ref_id=len(contacts), silent_failure=None, incident_id=None, label="late", true_order_id=o1))
    o2 = add_order(mid, NOW - timedelta(days=4), "electronics", False, dict(delay=-20, silent=False, items=[("ELEC-HEADPHONES", 129.0)], tracking_views=1, warehouse="W2", carrier="B"))
    add_order(mid, NOW - timedelta(days=30), "groceries", False, dict(delay=-10, silent=False, tracking_views=0, warehouse="W1", carrier="A"))
    add_order(mid, NOW - timedelta(days=75), "home", False, dict(delay=-15, silent=False, tracking_views=0, warehouse="W2", carrier="B"))

    # ---------------- subscriptions ----------------
    subs = []
    for c in cust:
        if c["id"] == mid or rng.random() > 0.40: continue
        last = NOW - timedelta(days=float(rng.uniform(60, 200) if rng.random() < 0.10 else rng.uniform(0, 45)))
        subs.append(dict(id=len(subs) + 1, customer_id=c["id"], name=str(rng.choice(SUBS)), monthly_price=round(float(rng.uniform(4.99, 19.99)), 2),
                         started_at=iso(NOW - timedelta(days=float(rng.uniform(80, 700)))), last_used_at=iso(last)))
    subs.append(dict(id=len(subs) + 1, customer_id=mid, name="Streaming add-on", monthly_price=9.99,
                     started_at=iso(NOW - timedelta(days=200)), last_used_at=iso(NOW - timedelta(days=70))))

    # ---------------- rewards / points / redemptions / history ----------------
    chart = []
    rid = 0
    for (name, c1, c2, cash) in REWARDS: rid += 1; chart.append(dict(id=rid, effective_at=iso(NOW - timedelta(days=400)), reward_name=name, points_cost=c1, cash_value=cash))
    for (name, c1, c2, cash) in REWARDS: rid += 1; chart.append(dict(id=rid, effective_at=iso(NOW - timedelta(days=20)), reward_name=name, points_cost=c2, cash_value=cash))
    points, redemptions, phist, gt_p = [], [], [], []
    n_reward = len(REWARDS)
    for c in cust:
        cid = c["id"]
        u = float(rng.beta(2, 3))                       # latent redemption propensity
        lam = 0.004 + 0.06 * u                           # per-day redemption hazard
        pref = int(rng.integers(n_reward))
        nred = int(rng.poisson(u * 7))
        for _ in range(nred):
            r = pref if rng.random() < 0.7 else int(rng.integers(n_reward))
            redemptions.append(dict(customer_id=cid, reward_name=REWARDS[r][0], points=REWARDS[r][1], redeemed_at=iso(NOW - timedelta(days=float(rng.uniform(1, 350))))))
        bal = int(np.clip(rng.lognormal(np.log(450), 0.9), 20, 6000))
        if rng.random() < 0.15:
            bal = max(bal, int(rng.integers(500, 3000))); dte = float(rng.uniform(0.5, 7))
        elif rng.random() < 0.03:
            bal = int(rng.integers(30, 480)); dte = float(rng.uniform(0.5, 7))
        else:
            dte = float(rng.uniform(8, 365))
        points.append(dict(customer_id=cid, balance=bal, expires_at=iso(NOW + timedelta(days=dte))))
        Tc = float(rng.exponential(1 / lam))
        gt_p.append(dict(kind="points", ref_id=cid, will_redeem=int(Tc < dte)))
        for cyc in range(2):                              # previous expiry cycles (observed history)
            win = float(rng.uniform(30, 120)); Th = float(rng.exponential(1 / lam))
            phist.append(dict(customer_id=cid, cycle=cyc, balance=int(np.clip(rng.lognormal(np.log(450), 0.9), 20, 6000)), window_days=win,
                              redeemed=int(Th < win), days_to_redeem=min(Th, win)))
    # Maya's points: 1,800 expiring in 3 days, usually redeems an Iced latte
    for p in points:
        if p["customer_id"] == mid: p.update(balance=1800, expires_at=iso(NOW + timedelta(days=3)))
    redemptions = [r for r in redemptions if r["customer_id"] != mid]
    for k in range(5):
        redemptions.append(dict(customer_id=mid, reward_name="Iced latte" if k < 4 else "Pastry of the day", points=150 if k < 4 else 120, redeemed_at=iso(NOW - timedelta(days=30 * (k + 1)))))
    phist = [h for h in phist if h["customer_id"] != mid] + [dict(customer_id=mid, cycle=c_, balance=900, window_days=90.0, redeemed=0, days_to_redeem=90.0) for c_ in range(2)]
    for g in gt_p:
        if g["ref_id"] == mid: g["will_redeem"] = 0

    # ---------------- customer ground truth (hidden response function) ----------------
    o_df = pd.DataFrame(orders)
    avg_amt = o_df.groupby("customer_id").amount.mean().to_dict()
    cnt = o_df.groupby("customer_id").size().to_dict()
    for c in cust:
        cid, s = c["id"], c["segment"]
        tau = SEG_TAU[s] * float(rng.gamma(2, 0.5))
        if rng.random() < 0.25: tau *= 0.1
        if avg_amt.get(cid, 40) < 35: tau *= 1.3
        tau = float(np.clip(tau, -0.01, 0.2))
        if rng.random() < 0.04: tau = -0.01
        base = float(np.clip(SEG_BASE[s] + rng.normal(0, 0.05), 0.2, 0.97))
        spend = max(20.0, cnt.get(cid, 2) / 365 * 90 * avg_amt.get(cid, 40) * float(rng.uniform(0.8, 1.2)))
        gt_c.append(dict(kind="customer", ref_id=cid, is_ring=int(cid in ring_of), ring_id=ring_of.get(cid), true_response=tau,
                         backfire=int(rng.random() < 0.08), base_retention=base, spend_90d=spend))
    gt_c[0].update(true_response=0.08, base_retention=0.75, backfire=0)

    # ---------------- NLP training set (agent-tagged historical tickets) ----------------
    rng2 = np.random.default_rng(1234)
    nlp_rows = []
    for _ in range(1600):
        k = TYPES[int(rng2.choice(len(TYPES), p=[.2, .15, .1, .12, .1, .13, .2]))]
        nlp_rows.append(dict(text=make_text(rng2, k, str(rng2.choice(WAREHOUSES)), str(rng2.choice(CARRIERS)), "Dec 03", 0.3), label=k))

    # ---------------- write DB ----------------
    with eng.begin() as conn:
        for t in ["customers", "orders", "order_items", "price_events", "subscriptions", "points_accounts", "redemptions",
                  "points_history", "reward_chart_versions", "contacts", "_ground_truth", "nlp_training"]:
            conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
    pd.DataFrame(cust).to_sql("customers", eng, index=False)
    pd.DataFrame(orders).to_sql("orders", eng, index=False)
    pd.DataFrame(items).to_sql("order_items", eng, index=False)
    pd.DataFrame(ev_rows, columns=["sku", "price", "effective_at"]).rename_axis("id").reset_index().to_sql("price_events", eng, index=False)
    pd.DataFrame(subs).to_sql("subscriptions", eng, index=False)
    pd.DataFrame(points).to_sql("points_accounts", eng, index=False)
    pd.DataFrame(redemptions).to_sql("redemptions", eng, index=False)
    pd.DataFrame(phist).to_sql("points_history", eng, index=False)
    pd.DataFrame(chart).to_sql("reward_chart_versions", eng, index=False)
    cdf = pd.DataFrame(contacts); cdf.insert(0, "id", range(1, len(cdf) + 1))
    drop = (rng.random(len(cdf)) < 0.40) & (cdf.customer_id != mid)      # 40% of tickets arrive WITHOUT an order reference
    cdf["order_id"] = cdf.order_id.astype("Float64").mask(drop)
    cdf.to_sql("contacts", eng, index=False)
    pd.concat([pd.DataFrame(gt_c), pd.DataFrame(gt_o), pd.DataFrame(gt_p)], ignore_index=True).to_sql("_ground_truth", eng, index=False)
    pd.DataFrame(nlp_rows).to_sql("nlp_training", eng, index=False)
    db.init_runtime_tables(eng)
    load_contracts_yaml(eng)
    if verbose:
        print(f"seeded: {len(cust)} customers, {len(orders)} orders, {len(items)} items, {len(contacts)} contacts, {len(ring_of)} ring members")
    return eng


def load_contracts_yaml(eng, path=None, only_active=None):
    cfg = yaml.safe_load(open(path or config.CONTRACTS_YAML, encoding="utf8"))["contracts"]
    with eng.begin() as c:
        c.execute(text("DELETE FROM promise_contracts"))
        for k, v in cfg.items():
            c.execute(text("INSERT INTO promise_contracts VALUES(:i,:t,:p,:a,:e)"),
                      dict(i=k, t=k, p=json.dumps(v["params"]), a=int(v["active"] and (only_active is None or k not in CORE or k in only_active)), e=int(v["experimental"])))


if __name__ == "__main__":
    build(verbose=True)

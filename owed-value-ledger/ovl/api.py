"""FastAPI app. OpenAPI docs at /docs. Serves the single-page UI at /."""
import json
from datetime import timedelta

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

from . import config, db, engine
from .abuse import guard
from .eval import cards as cards_mod
from .policy import settlement as pol

app = FastAPI(title="Owed-Value Ledger", description="What the BRAND owes the CUSTOMER, settled before they ask. SYNTHETIC DATA DEMO.", version="1.0")
eng = db.get_engine()
NOW = config.NOW
FMT = "%Y-%m-%d %H:%M:%S"


def q(sql, **p):
    with eng.connect() as c:
        return [dict(r._mapping) for r in c.execute(text(sql), p)]


def q_one(sql, **p):
    with eng.connect() as c:
        return c.execute(text(sql), p).scalar()


def ts(t):
    return t.strftime(FMT)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


# ---------------- customer ----------------
def entry_view(r):
    r = dict(r)
    r["reasons"] = json.loads(r["reasons"]) if r.get("reasons") else []
    r["pending"] = r["status"] in ("open", "review")
    # silent credits simply appear: they carry the factual note but no alert flag
    r["alert"] = r["settlement_mode"] == "explicit" and r["status"] == "settled"
    return r


@app.get("/customers")
def customers(search: str = Query("", alias="q"), segment: str = "", filter: str = "", offset: int = 0, limit: int = 15):
    """Search (id or name), filter and paginate over ALL customers.
    filter: with_entries | review | holdout | no_entries | flagged"""
    where, p = ["1=1"], {}
    term = search.strip()
    if term:
        if term.lstrip("#").isdigit():
            where.append("(c.id=:qid OR c.name LIKE :qn)"); p["qid"] = int(term.lstrip("#")); p["qn"] = f"%{term}%"
        else:
            where.append("c.name LIKE :qn"); p["qn"] = f"%{term}%"
    if segment in ("new", "regular", "vip"):
        where.append("c.segment=:seg"); p["seg"] = segment
    where.append({"with_entries": "COALESCE(l.entries,0)>0", "no_entries": "COALESCE(l.entries,0)=0", "review": "COALESCE(l.held,0)>0",
                  "holdout": "COALESCE(l.hold,0)=1", "flagged": "c.id IN (SELECT customer_id FROM abuse_flags)"}.get(filter, "1=1"))
    base = ("FROM customers c LEFT JOIN (SELECT customer_id, COUNT(*) entries, SUM(CASE WHEN status='settled' THEN value_usd ELSE 0 END) returned, "
            "SUM(CASE WHEN status='review' THEN 1 ELSE 0 END) held, MAX(CASE WHEN experiment_arm='holdout' THEN 1 ELSE 0 END) hold "
            "FROM ledger_entries GROUP BY customer_id) l ON l.customer_id=c.id WHERE " + " AND ".join(where))
    total = q_one(f"SELECT COUNT(*) n {base}", **p)
    rows = q(f"SELECT c.id, c.name, c.segment, COALESCE(l.entries,0) entries, ROUND(COALESCE(l.returned,0),2) returned, COALESCE(l.held,0) held, "
             f"COALESCE(l.hold,0) holdout {base} ORDER BY (c.id=1) DESC, (COALESCE(l.held,0)>0) ASC, entries DESC, c.id LIMIT :lim OFFSET :off", lim=max(1, min(limit, 100)), off=max(0, offset), **p)
    return dict(total=total, items=rows, offset=offset)


@app.get("/customers/{cid}/ledger")
def ledger(cid: int):
    c = q("SELECT id,name,segment FROM customers WHERE id=:i", i=cid)
    if not c:
        raise HTTPException(404, "unknown customer")
    rows = q("SELECT * FROM ledger_entries WHERE customer_id=:i ORDER BY event_at DESC", i=cid)
    return dict(customer=c[0], entries=[entry_view(r) for r in rows],
                lifetime_value_returned=round(sum(r["value_usd"] for r in rows if r["status"] == "settled"), 2),
                open_or_pending=round(sum(r["value_usd"] for r in rows if r["status"] in ("open", "review")), 2))


@app.get("/customers/{cid}/summary")
def summary(cid: int):
    L = ledger(cid)
    settled = [e for e in L["entries"] if e["status"] == "settled"]
    by = {}
    for e in settled:
        by[e["contract_type"]] = round(by.get(e["contract_type"], 0) + e["value_usd"], 2)
    return dict(customer=L["customer"], we_owe_you_total=L["lifetime_value_returned"], settled_items=len(settled),
                pending_items=len([e for e in L["entries"] if e["pending"]]), by_contract=by, contacts_needed=0)


# ---------------- run / simulate ----------------
@app.post("/run")
def run(customer_id: int | None = None):
    only = {customer_id} if customer_id else None
    r = engine.run_pipeline(eng, NOW, only=only, realtime=bool(customer_id))
    refresh_abuse()
    return dict(detected=r["detected"], new_candidates=r["new"], inserted=r["inserted"], incidents=[i["id"] for i in r["incidents"]])


class EventIn(BaseModel):
    customer_id: int = config.MAYA_ID
    kind: str = "late_order"        # late_order | price_drop


@app.post("/simulate/event")
def simulate_event(ev: EventIn):
    if ev.kind not in ("late_order", "price_drop"):
        raise HTTPException(400, "kind must be late_order or price_drop")
    with eng.begin() as c:
        oid = (c.execute(text("SELECT MAX(id) FROM orders")).scalar() or 0) + 1
        if ev.kind == "late_order":
            promised, delivered, placed = NOW - timedelta(hours=2), NOW - timedelta(minutes=5), NOW - timedelta(hours=8)
            amt = 48.20
            c.execute(text("INSERT INTO orders VALUES(:id,:c,:p,:pr,:d,:a,'groceries','delivered','A','W1',2)"),
                      dict(id=oid, c=ev.customer_id, p=ts(placed), pr=ts(promised), d=ts(delivered), a=amt))
            c.execute(text("INSERT INTO order_items VALUES(:o,'GRO-DEMO',:a)"), dict(o=oid, a=amt))
            cid_ = (c.execute(text("SELECT COALESCE(MAX(id),0)+1 FROM contacts")).scalar())
            c.execute(text("INSERT INTO contacts(id,customer_id,created_at,channel,text,incident_id,order_id) VALUES(:i,:c,:t,'chat',:x,NULL,:o)"),
                      dict(i=cid_, c=ev.customer_id, t=ts(NOW - timedelta(minutes=40)), x="Still waiting for my order, it was due two hours ago", o=oid))
        else:
            sku = f"DEMO-{oid}"
            placed = NOW - timedelta(days=3)
            c.execute(text("INSERT INTO orders VALUES(:id,:c,:p,:pr,:d,:a,'electronics','delivered','B','W2',0)"),
                      dict(id=oid, c=ev.customer_id, p=ts(placed), pr=ts(placed + timedelta(days=1)), d=ts(placed + timedelta(days=1)), a=89.0))
            c.execute(text("INSERT INTO order_items VALUES(:o,:s,89.0)"), dict(o=oid, s=sku))
            c.execute(text("INSERT INTO price_events(id,sku,price,effective_at) VALUES((SELECT COALESCE(MAX(id),0)+1 FROM price_events),:s,89.0,:t)"), dict(s=sku, t=ts(placed - timedelta(days=30))))
            c.execute(text("INSERT INTO price_events(id,sku,price,effective_at) VALUES((SELECT COALESCE(MAX(id),0)+1 FROM price_events),:s,74.0,:t)"), dict(s=sku, t=ts(NOW - timedelta(hours=1))))
    before = {r["id"] for r in q("SELECT id FROM ledger_entries WHERE customer_id=:c", c=ev.customer_id)}
    engine.run_pipeline(eng, NOW, only={ev.customer_id}, realtime=True)
    new = [entry_view(r) for r in q("SELECT * FROM ledger_entries WHERE customer_id=:c ORDER BY id", c=ev.customer_id) if r["id"] not in before]
    return dict(order_id=oid, new_entries=new)


# ---------------- ops ----------------
_ab = {"df": None}


def refresh_abuse():
    cust = pd.read_sql("SELECT * FROM customers", eng)
    claims, payout = engine.claims_from_ledger(eng, NOW)
    ap = db.load_contract_rows(eng)["abuse_guard"]["params"]
    _ab["df"] = guard.score(cust, claims, payout, ap, NOW)
    _ab["cust"] = cust


@app.get("/ops/overview")
def overview():
    k = q("SELECT status, COUNT(*) n, ROUND(SUM(value_usd),2) usd FROM ledger_entries GROUP BY status")
    by = {r["status"]: r for r in k}
    arms = q("SELECT experiment_arm a, COUNT(*) n FROM ledger_entries GROUP BY a")
    tts = q("SELECT (julianday(settled_at)-julianday(event_at))*24 h FROM ledger_entries WHERE status='settled' AND experiment_arm='treatment'")
    hs = sorted(r["h"] for r in tts)
    trt = sum(r["n"] for r in q("SELECT COUNT(*) n FROM ledger_entries WHERE experiment_arm='treatment'"))
    settled_trt = q("SELECT COUNT(*) n FROM ledger_entries WHERE experiment_arm='treatment' AND status='settled'")[0]["n"]
    types = q("SELECT contract_type t, COUNT(*) n, ROUND(SUM(CASE WHEN status='settled' THEN value_usd ELSE 0 END),2) settled FROM ledger_entries GROUP BY t ORDER BY settled DESC")
    return dict(open_liability_usd=round((by.get("open", {}).get("usd") or 0) + (by.get("review", {}).get("usd") or 0), 2),
                settled_value_usd=by.get("settled", {}).get("usd") or 0, settled_entries=by.get("settled", {}).get("n", 0),
                review_queue=by.get("review", {}).get("n", 0), claim_pending_holdout=by.get("open", {}).get("n", 0),
                coverage=settled_trt / max(1, trt), median_time_to_settle_h=hs[len(hs) // 2] if hs else None,
                abuse_flags=q("SELECT COUNT(*) n FROM abuse_flags")[0]["n"], by_contract=types, arms={r["a"]: r["n"] for r in arms})


def _review_where(contract, reason):
    w, p = ["l.status='review'"], {}
    if contract:
        w.append("l.contract_type=:ct"); p["ct"] = contract
    if reason:
        w.append("l.review_reason=:rs"); p["rs"] = reason
    return " AND ".join(w), p


@app.get("/ops/review-summary")
def review_summary():
    """What is REALLY in the queue (not just its most valuable rows)."""
    by_c = q("SELECT contract_type t, COUNT(*) n, ROUND(SUM(value_usd),2) usd FROM ledger_entries WHERE status='review' GROUP BY t ORDER BY n DESC")
    by_r = q("SELECT review_reason r, COUNT(*) n FROM ledger_entries WHERE status='review' GROUP BY r ORDER BY n DESC")
    return dict(entries=sum(x["n"] for x in by_c), customers=q_one("SELECT COUNT(DISTINCT customer_id) FROM ledger_entries WHERE status='review'"),
                value_usd=round(sum(x["usd"] for x in by_c), 2), by_contract=by_c, by_reason=by_r)


@app.get("/ops/review-queue")
def review_queue(contract: str = "", reason: str = "", sort: str = "risk", offset: int = 0, limit: int = 25):
    w, p = _review_where(contract, reason)
    order = "abuse_score DESC, l.value_usd DESC" if sort == "risk" else "l.value_usd DESC"
    total = q_one(f"SELECT COUNT(*) FROM ledger_entries l WHERE {w}", **p)
    rows = q("SELECT l.id, l.customer_id, c.name, l.contract_type, l.value_usd, l.review_reason, l.note, l.reasons, "
             "COALESCE((SELECT MAX(score) FROM abuse_flags a WHERE a.customer_id=l.customer_id),0) abuse_score "
             f"FROM ledger_entries l JOIN customers c ON c.id=l.customer_id WHERE {w} ORDER BY {order} LIMIT :lim OFFSET :off",
             lim=max(1, min(limit, 200)), off=max(0, offset), **p)
    return dict(total=total, items=rows, offset=offset, sort=sort)


@app.get("/ops/review-cases")
def review_cases(contract: str = "", reason: str = "", offset: int = 0, limit: int = 10):
    """One row per CUSTOMER (a case), riskiest first, with the full mix of what is being held."""
    w, p = _review_where(contract, reason)
    total = q_one(f"SELECT COUNT(DISTINCT l.customer_id) FROM ledger_entries l WHERE {w}", **p)
    cases = q("SELECT l.customer_id, c.name, c.segment, COUNT(*) entries, ROUND(SUM(l.value_usd),2) value_usd, MAX(l.review_reason) reason, "
              "COALESCE((SELECT MAX(score) FROM abuse_flags a WHERE a.customer_id=l.customer_id),0) abuse_score "
              f"FROM ledger_entries l JOIN customers c ON c.id=l.customer_id WHERE {w} GROUP BY l.customer_id "
              "ORDER BY abuse_score DESC, value_usd DESC LIMIT :lim OFFSET :off", lim=max(1, min(limit, 50)), off=max(0, offset), **p)
    for c in cases:
        ents = q("SELECT id, contract_type, value_usd, note FROM ledger_entries WHERE customer_id=:c AND status='review' ORDER BY value_usd DESC", c=c["customer_id"])
        c["entries_list"] = ents
        mix = {}
        for e in ents:
            mix[e["contract_type"]] = mix.get(e["contract_type"], 0) + 1
        c["mix"] = mix
    return dict(total=total, items=cases, offset=offset)


@app.post("/ops/review-case/{cid}/approve")
def approve_case(cid: int):
    return _review_case(cid, "settled")


@app.post("/ops/review-case/{cid}/deny")
def deny_case(cid: int):
    return _review_case(cid, "denied")


def _review_case(cid, to):
    """Decide every entry in a customer's review case. A person makes this call; nothing is auto-denied."""
    with eng.begin() as c:
        n = c.execute(text("UPDATE ledger_entries SET status=:s, settled_at=:t WHERE customer_id=:c AND status='review'"),
                      dict(s=to, t=ts(NOW) if to == "settled" else None, c=cid)).rowcount
    if not n:
        raise HTTPException(409, "no entries of this customer are in review")
    return dict(customer_id=cid, decided=n, status=to)


def _review(eid, to):
    with eng.begin() as c:
        row = c.execute(text("SELECT status FROM ledger_entries WHERE id=:i"), dict(i=eid)).fetchone()
        if not row:
            raise HTTPException(404, "unknown entry")
        if row[0] != "review":
            raise HTTPException(409, "only entries in review can be decided (denials are never automatic)")
        c.execute(text("UPDATE ledger_entries SET status=:s, settled_at=:t WHERE id=:i"), dict(s=to, t=ts(NOW) if to == "settled" else None, i=eid))
    return dict(id=eid, status=to)


@app.post("/ops/review/{eid}/approve")
def approve(eid: int):
    return _review(eid, "settled")


@app.post("/ops/review/{eid}/deny")
def deny(eid: int):
    return _review(eid, "denied")


@app.get("/ops/graph")
def graph():
    if _ab["df"] is None:
        refresh_abuse()
    df, cust = _ab["df"], _ab["cust"]
    ring_comps = df[df.ring >= 0.5].comp.unique()
    benign = [c for c in df[(df.comp_size >= 2) & (df.ring < 0.5)].comp.unique()][:6]
    ids = df[df.comp.isin(list(ring_comps) + benign)].index
    nodes = [dict(id=int(i), flagged=bool(df.loc[i, "flagged"]), score=round(float(df.loc[i, "abuse_score"]), 2), comp=int(df.loc[i, "comp"]),
                  claims=int(df.loc[i, "claims30"]), ring=bool(df.loc[i, "ring"] >= 0.5)) for i in ids]
    return dict(nodes=nodes, edges=guard.graph_edges(cust, set(ids)), rings=int(len(ring_comps)))


@app.get("/ops/metrics")
def metrics():
    if not config.METRICS_PATH.exists():
        raise HTTPException(404, "metrics.json missing - run `python run.py`")
    return json.loads(config.METRICS_PATH.read_text(encoding="utf8"))


@app.get("/ops/olist")
def olist():
    from .olist import analyze
    if not analyze.OUT.exists():
        raise HTTPException(404, "Real-data analysis not run: put the Olist CSVs in data/olist/ and run `python -m ovl.olist.analyze`")
    return json.loads(analyze.OUT.read_text(encoding="utf8"))


def _real(name):
    import importlib
    mod = importlib.import_module(f"ovl.realdata.{name}")
    if not mod.OUT.exists():
        raise HTTPException(404, f"{name} analysis not run: place the data under data/ and run `python -m ovl.realdata.{name}`")
    return json.loads(mod.OUT.read_text(encoding="utf8"))


@app.get("/ops/dunnhumby")
def dunnhumby():
    return _real("dunnhumby")


@app.get("/ops/twcs")
def twcs():
    return _real("twcs")


@app.get("/ops/models")
def models():
    return cards_mod.build_cards(metrics())


# ---------------- contracts ----------------
@app.get("/contracts")
def contracts():
    return list(db.load_contract_rows(eng).values())


class ContractIn(BaseModel):
    params: dict | None = None
    active: bool | None = None


@app.put("/contracts/{cid}")
def put_contract(cid: str, body: ContractIn):
    rows = db.load_contract_rows(eng)
    if cid not in rows:
        raise HTTPException(404, "unknown contract")
    r = rows[cid]
    params = {**r["params"], **(body.params or {})}
    active = r["active"] if body.active is None else body.active
    with eng.begin() as c:
        c.execute(text("UPDATE promise_contracts SET params_json=:p, active=:a WHERE id=:i"), dict(p=json.dumps(params), a=int(active), i=cid))
    rows = db.load_contract_rows(eng)       # mirror back to YAML so the file stays the human-readable registry
    out = {k: dict(active=v["active"], experimental=v["experimental"], params=v["params"]) for k, v in rows.items()}
    (config.DATA_DIR / "contracts.live.yaml").write_text("# Live registry mirrored from the DB (config/contracts.yaml is the seed).\n" +yaml.safe_dump({"contracts": out}, sort_keys=False), encoding="utf8")
    return rows[cid]


@app.get("/health")
def health():
    return dict(ok=True, clock=str(NOW), llm=config.LLM_ENABLED)

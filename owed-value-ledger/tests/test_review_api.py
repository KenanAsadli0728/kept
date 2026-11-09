from fastapi.testclient import TestClient


def client(world):
    from ovl.api import app
    return TestClient(app)


def test_ledger_and_summary(world):
    c = client(world)
    s = c.get("/customers/1/summary").json()
    assert s["we_owe_you_total"] > 40 and s["contacts_needed"] == 0
    L = c.get("/customers/1/ledger").json()
    assert L["entries"] and all("reasons" in e for e in L["entries"])
    assert c.get("/customers/999999/ledger").status_code == 404


def test_denials_are_never_automatic_only_from_review(world):
    c = client(world)
    settled = [e for e in c.get("/customers/1/ledger").json()["entries"] if e["status"] == "settled"][0]
    assert c.post(f"/ops/review/{settled['id']}/deny").status_code == 409      # cannot deny a settled entry
    q = c.get("/ops/review-queue?limit=5").json()["items"]
    assert len(q) >= 2
    assert c.post(f"/ops/review/{q[0]['id']}/deny").json()["status"] == "denied"
    assert c.post(f"/ops/review/{q[1]['id']}/approve").json()["status"] == "settled"
    assert c.post(f"/ops/review/{q[1]['id']}/approve").status_code == 409     # already decided


def test_simulate_event_creates_live_iou_idempotently(world):
    c = client(world)
    r = c.post("/simulate/event", json={"customer_id": 1, "kind": "late_order"}).json()
    assert len(r["new_entries"]) == 1 and r["new_entries"][0]["status"] == "settled"
    n = len(c.get("/customers/1/ledger").json()["entries"])
    assert c.post("/run", params={"customer_id": 1}).json()["inserted"] == 0
    assert len(c.get("/customers/1/ledger").json()["entries"]) == n
    assert c.post("/simulate/event", json={"customer_id": 1, "kind": "nope"}).status_code == 400


def test_price_drop_event_is_silent_for_unaware_customer(world):
    c = client(world)
    r = c.post("/simulate/event", json={"customer_id": 1, "kind": "price_drop"}).json()
    e = r["new_entries"][0]
    assert e["contract_type"] == "price_protection" and e["settlement_mode"] == "silent"


def test_contract_edit_roundtrip(world):
    c = client(world)
    cur = {x["id"]: x for x in c.get("/contracts").json()}
    new = c.put("/contracts/price_protection", json={"params": {"window_days": 21}}).json()
    assert new["params"]["window_days"] == 21 and new["params"]["min_refund"] == cur["price_protection"]["params"]["min_refund"]
    c.put("/contracts/price_protection", json={"params": {"window_days": 14}})
    assert c.put("/contracts/nope", json={}).status_code == 404


def test_overview_and_models_endpoints(world):
    from ovl.eval import simulate
    simulate.compute_metrics(world["eng"], models=world["models"])
    c = client(world)
    o = c.get("/ops/overview").json()
    assert o["settled_entries"] > 0 and 0 <= o["coverage"] <= 1
    cards = c.get("/ops/models").json()
    assert len(cards) >= 8 and all(x["limits"] and x["metrics"] for x in cards)
    assert "SYNTHETIC" in c.get("/ops/metrics").json()["disclaimer"]
    assert c.get("/ops/graph").json()["nodes"]

import collections
from fastapi.testclient import TestClient


def client():
    from kept.api import app
    return TestClient(app)


def test_customer_list_covers_everyone_and_paginates(world):
    c = client()
    first = c.get("/customers", params=dict(limit=15)).json()
    total = first["total"]
    assert total == 1500 and len(first["items"]) == 15 and first["items"][0]["id"] == 1          # Maya first
    seen, off = set(), 0
    while off < total:
        page = c.get("/customers", params=dict(offset=off, limit=100)).json()["items"]
        seen |= {x["id"] for x in page}
        off += 100
    assert len(seen) == total                                                                    # every customer reachable


def test_customer_search_and_filters(world):
    c = client()
    assert [x["id"] for x in c.get("/customers", params=dict(q="#1")).json()["items"]][0] == 1
    assert c.get("/customers", params=dict(q="Maya")).json()["total"] == 1
    flagged = c.get("/customers", params=dict(filter="flagged")).json()
    review = c.get("/customers", params=dict(filter="review")).json()
    assert flagged["total"] > 0 and review["total"] > 0 and all(x["held"] > 0 for x in review["items"])
    none = c.get("/customers", params=dict(filter="no_entries", limit=50)).json()["items"]
    assert none and all(x["entries"] == 0 for x in none)
    assert all(x["segment"] == "vip" for x in c.get("/customers", params=dict(segment="vip", limit=50)).json()["items"])


def test_default_customer_page_is_not_all_abusers(world):
    items = client().get("/customers", params=dict(limit=12)).json()["items"]
    assert sum(1 for x in items if x["held"] > 0) <= 3                                           # ordinary customers come first


def test_review_summary_shows_real_composition(world):
    c = client()
    s = c.get("/ops/review-summary").json()
    assert s["entries"] == sum(x["n"] for x in s["by_contract"]) and s["customers"] > 0
    assert len(s["by_contract"]) >= 3


def test_review_queue_is_risk_sorted_not_value_sorted_and_filterable(world):
    c = client()
    q = c.get("/ops/review-queue", params=dict(limit=100)).json()
    scores = [x["abuse_score"] for x in q["items"]]
    assert scores == sorted(scores, reverse=True)
    page = collections.Counter(x["contract_type"] for x in q["items"][:60])
    overall = {x["t"]: x["n"] / sum(y["n"] for y in c.get("/ops/review-summary").json()["by_contract"]) for x in c.get("/ops/review-summary").json()["by_contract"]}
    n = sum(page.values())
    assert max(abs(page.get(t, 0) / n - share) for t, share in overall.items()) < 0.25          # first page mirrors the queue's true mix
    ct = max(page, key=page.get)
    f = c.get("/ops/review-queue", params=dict(contract=ct, limit=200)).json()
    assert f["total"] > 0 and all(x["contract_type"] == ct for x in f["items"])
    assert c.get("/ops/review-queue", params=dict(offset=q["total"] + 5)).json()["items"] == []


def test_review_cases_group_by_customer_and_decide_whole_case(world):
    c = client()
    k = c.get("/ops/review-cases", params=dict(limit=5)).json()
    assert k["total"] > 0 and all(x["entries"] == len(x["entries_list"]) == sum(x["mix"].values()) for x in k["items"])
    cid = k["items"][-1]["customer_id"]
    r = c.post(f"/ops/review-case/{cid}/approve").json()
    assert r["decided"] > 0 and r["status"] == "settled"
    assert c.post(f"/ops/review-case/{cid}/deny").status_code == 409                             # already decided: no second decision

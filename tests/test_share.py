from fastapi.testclient import TestClient

HTML = {"accept": "text/html"}


def client():
    from kept.api import app
    return TestClient(app, follow_redirects=False)


def test_no_pin_means_nothing_changes(world, monkeypatch):
    monkeypatch.delenv("KEPT_PIN", raising=False)
    c = client()
    assert c.get("/customers", params=dict(limit=1)).status_code == 200
    assert c.get("/", headers=HTML).status_code == 200


def test_pin_blocks_api_and_ui_until_correct(world, monkeypatch):
    monkeypatch.setenv("KEPT_PIN", "4821")
    c = client()
    r = c.get("/customers", params=dict(limit=1))
    assert r.status_code == 401 and r.json()["detail"] == "PIN required"
    page = c.get("/", headers=HTML)
    assert page.status_code == 401 and "Enter the access PIN" in page.text and "Wrong PIN" not in page.text
    wrong = c.get("/", params=dict(pin="0000"), headers=HTML)
    assert wrong.status_code == 401 and "Wrong PIN" in wrong.text
    assert c.post("/ops/review-case/1/approve").status_code == 401                 # writes are blocked too


def test_correct_pin_sets_cookie_and_unlocks(world, monkeypatch):
    monkeypatch.setenv("KEPT_PIN", "4821")
    c = client()
    ok = c.get("/", params=dict(pin="4821"), headers=HTML)
    assert ok.status_code == 303 and "kept_pin=" in ok.headers["set-cookie"] and "httponly" in ok.headers["set-cookie"].lower()
    assert "4821" not in ok.headers["set-cookie"]                                   # the PIN itself is not stored in the cookie
    cookie = ok.cookies.get("kept_pin")
    assert c.get("/customers", params=dict(limit=1), cookies={"kept_pin": cookie}).status_code == 200
    assert c.get("/customers", params=dict(limit=1), cookies={"kept_pin": "forged"}).status_code == 401


def test_installability_files_stay_public_behind_the_pin(world, monkeypatch):
    monkeypatch.setenv("KEPT_PIN", "4821")
    c = client()
    for path in ("/manifest.webmanifest", "/sw.js", "/icons/icon-192.png", "/health"):
        assert c.get(path).status_code == 200, path                                  # browsers fetch these without cookies


def test_share_info_reports_state(world, monkeypatch):
    monkeypatch.delenv("KEPT_SHARE", raising=False)
    monkeypatch.delenv("KEPT_PIN", raising=False)
    off = client().get("/ops/share").json()
    assert off["shared"] is False and off["urls"] == [] and "--share" in off["note"]
    monkeypatch.setenv("KEPT_SHARE", "1")
    monkeypatch.setenv("KEPT_PIN", "1")
    on = client().get("/ops/share", headers={"cookie": "kept_pin=" + __import__("hashlib").sha256(b"kept-demo-gate:1").hexdigest()}).json()
    assert on["shared"] is True and on["pin_enabled"] is True
    assert all(u.startswith("http://") and not u.startswith("http://127.") for u in on["urls"])

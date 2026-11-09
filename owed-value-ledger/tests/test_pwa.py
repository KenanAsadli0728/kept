import json
import re
import struct
from pathlib import Path

from fastapi.testclient import TestClient

STATIC = Path(__file__).resolve().parent.parent / "ovl" / "static"


def client():
    from ovl.api import app
    return TestClient(app)


def test_manifest_is_valid_and_installable(world):
    r = client().get("/manifest.webmanifest")
    assert r.status_code == 200 and "manifest+json" in r.headers["content-type"]
    m = r.json()
    for k in ("name", "short_name", "start_url", "scope", "display", "icons", "theme_color", "background_color"):
        assert k in m, k
    assert m["display"] == "standalone" and m["start_url"] == "/"
    sizes = {(i["sizes"], i.get("purpose", "any")) for i in m["icons"]}
    assert ("192x192", "any") in sizes and ("512x512", "any") in sizes and ("512x512", "maskable") in sizes   # Chrome installability


def test_every_manifest_icon_is_served_and_is_a_real_image(world):
    c = client()
    for icon in json.loads((STATIC / "manifest.webmanifest").read_text(encoding="utf8"))["icons"]:
        r = c.get(icon["src"])
        assert r.status_code == 200, icon["src"]
        if icon["type"] == "image/png":
            assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
            w, h = struct.unpack(">II", r.content[16:24])
            assert f"{w}x{h}" == icon["sizes"]
        else:
            assert b"<svg" in r.content


def test_icon_route_rejects_path_traversal(world):
    c = client()
    for bad in ("..%2f..%2fconfig%2fcontracts.yaml", "nope.png", "../run.py", "ICON.PNG"):
        assert c.get(f"/icons/{bad}").status_code == 404


def test_service_worker_is_root_scoped_and_not_cached(world):
    r = client().get("/sw.js")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]
    assert r.headers["service-worker-allowed"] == "/" and "no-cache" in r.headers["cache-control"]


def test_service_worker_never_caches_live_data_or_writes():
    sw = (STATIC / "sw.js").read_text(encoding="utf8")
    live = re.search(r"const LIVE = /(.+)/;", sw).group(1)
    for path in ("/customers", "/customers/1/ledger", "/ops/overview", "/ops/review-cases", "/contracts", "/run", "/simulate/event", "/health", "/docs"):
        assert re.match(live, path + "?x=1") or re.match(live, path), path           # every API path is excluded from the cache
    assert "req.method !== 'GET'" in sw                                               # approvals / denials / runs are never intercepted
    assert not re.match(live, "/")                                                    # only the shell is cacheable


def test_index_links_the_app_and_still_serves_the_ui(world):
    r = client().get("/")
    assert r.status_code == 200
    html = r.text
    assert 'rel="manifest"' in html and 'name="theme-color"' in html and 'rel="apple-touch-icon"' in html
    assert "serviceWorker" in html and 'id="install"' in html and 'id="main"' in html


def test_existing_api_unchanged_by_pwa(world):
    c = client()
    assert c.get("/health").json()["ok"] is True
    assert c.get("/customers/1/summary").json()["contacts_needed"] == 0
    assert c.get("/customers", params=dict(limit=3)).json()["total"] > 0

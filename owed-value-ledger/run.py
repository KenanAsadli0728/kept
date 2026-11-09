"""One command: seed -> train -> run pipeline -> metrics -> serve.
   python run.py [--no-serve] [--port 8000] [--serve-only] [--open | --app]"""
import argparse
import os
import subprocess
import threading
import time
import urllib.request
import warnings
import webbrowser

warnings.filterwarnings("ignore")


def build(verbose=True, wedge=False, refresh_real=False):
    from ovl import db, engine, seed
    from ovl.eval import simulate, training
    t0 = time.time()

    def step(name):
        if verbose: print(f"[{time.time() - t0:5.1f}s] {name}", flush=True)

    step("seeding synthetic data (5,000 customers, 12 months)")
    eng = seed.build(verbose=verbose)
    if wedge:
        seed.load_contracts_yaml(eng, only_active=seed.WEDGE)
        step(f"WEDGE preset: only {seed.WEDGE} are active")
    step("training models (NLP, silent-failure + calibration, points survival, uplift)")
    training.train_all(eng, verbose=False)
    step("running detection + settlement pipeline")
    r = engine.run_pipeline(eng)
    step(f"  {r['inserted']} ledger entries written")
    step("simulating 90 days post-settlement + computing metrics")
    simulate.compute_metrics(eng)
    from ovl.olist import analyze as olist
    from ovl.realdata import dunnhumby, twcs
    for name, mod in (("Olist (late delivery, silent failures, repeat purchase)", olist), ("Dunnhumby (price protection, coupon campaigns)", dunnhumby),
                      ("Twitter support (effort, incidents)", twcs)):
        if not mod.available():
            step(f"REAL data not found for {name}: skipped (synthetic demo unaffected)")
        elif mod.OUT.exists() and not refresh_real:
            step(f"REAL {name}: using cached results (run with --refresh-real to recompute)")
        else:
            step(f"analysing REAL {name}")
            mod.run(verbose=False)
    step("done")


def open_when_ready(url, as_app):
    """Wait for the server, then open it. --app uses an Edge/Chrome 'app window' (no tabs or address bar)."""
    def work():
        for _ in range(120):
            try:
                urllib.request.urlopen(url + "/health", timeout=1)
                break
            except Exception:
                time.sleep(1)
        if as_app:
            for exe in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                        r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"):
                if os.path.exists(exe):
                    subprocess.Popen([exe, f"--app={url}", "--window-size=1280,900"])
                    return
        webbrowser.open(url)
    threading.Thread(target=work, daemon=True).start()


def have_data():
    from ovl import config
    return config.DB_PATH.exists() and (config.MODELS_DIR / "bundle.joblib").exists() and config.METRICS_PATH.exists()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-serve", action="store_true")
    ap.add_argument("--refresh-real", action="store_true", help="recompute the real-data analyses (Twitter takes about 1 minute)")
    ap.add_argument("--wedge", action="store_true", help="activate only the contracts the economics support")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--serve-only", action="store_true", help="skip the build when data already exists (keeps your ledger state)")
    ap.add_argument("--share", action="store_true", help="let phones/other computers on your Wi-Fi open the app (binds 0.0.0.0 and prints the address)")
    ap.add_argument("--pin", default=os.environ.get("OVL_PIN", ""), help="require this access PIN (recommended with --share)")
    ap.add_argument("--open", action="store_true", help="open the app in your browser when ready")
    ap.add_argument("--app", action="store_true", help="open it in an app window (Edge/Chrome --app mode)")
    ap.add_argument("--host-all", action="store_true", help="bind 0.0.0.0 (Docker)")
    a = ap.parse_args()
    if a.serve_only and have_data():
        print("Existing data found: skipping the build (use without --serve-only to rebuild).", flush=True)
    else:
        build(wedge=a.wedge, refresh_real=a.refresh_real)
    if not a.no_serve:
        if a.open or a.app:
            open_when_ready(f"http://127.0.0.1:{a.port}", as_app=a.app)
        import uvicorn
        if a.pin:
            os.environ["OVL_PIN"] = a.pin
        if a.share or a.host_all:
            os.environ["OVL_SHARE"] = "1"
            a.host_all = True
            from ovl.api import _lan_ips
            import socket
            print("\nShared on your network. On a phone connected to the SAME Wi-Fi open:", flush=True)
            for n, ip in enumerate(_lan_ips()):
                print(f"   http://{ip}:{a.port}" + ("   <- try this first" if n == 0 else "   (virtual/other adapter)"), flush=True)
            print(f"   http://{socket.gethostname()}.local:{a.port}   (if the IP changes)", flush=True)
            print("   PIN protection: " + ("ON (enter the PIN on the phone)" if a.pin else "OFF - anyone on this network can use it; add --pin 1234"), flush=True)
            print("   First time: allow Python through Windows Firewall for PRIVATE networks when Windows asks.", flush=True)
        print(f"\nKept -> http://{'0.0.0.0' if a.host_all else '127.0.0.1'}:{a.port}   (API docs: /docs)\n", flush=True)
        uvicorn.run("ovl.api:app", host="0.0.0.0" if a.host_all else "127.0.0.1", port=a.port, log_level="warning")

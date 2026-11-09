"""One command: seed -> train -> run pipeline -> metrics -> serve.   python run.py [--no-serve] [--port 8000]"""
import argparse
import time
import warnings

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-serve", action="store_true")
    ap.add_argument("--refresh-real", action="store_true", help="recompute the real-data analyses (Twitter takes about 1 minute)")
    ap.add_argument("--wedge", action="store_true", help="activate only the contracts the economics support")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host-all", action="store_true", help="bind 0.0.0.0 (Docker)")
    a = ap.parse_args()
    build(wedge=a.wedge, refresh_real=a.refresh_real)
    if not a.no_serve:
        import uvicorn
        print(f"\nOwed-Value Ledger -> http://127.0.0.1:{a.port}   (API docs: /docs)\n", flush=True)
        uvicorn.run("ovl.api:app", host="0.0.0.0" if a.host_all else "127.0.0.1", port=a.port, log_level="warning")

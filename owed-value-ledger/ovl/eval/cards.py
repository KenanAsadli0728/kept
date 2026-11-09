"""Model cards for the ops 'Models' tab. Metrics are read from metrics.json so they never go stale."""
import json

from .. import config, llm


def build_cards(M):
    m = M.get("models", {})
    nlp, sf, pts, inc = m.get("nlp", {}), m.get("silent_failure", {}), m.get("points", {}), m.get("incidents", {})
    ab, up, op, nt = M.get("abuse", {}), M.get("uplift", {}), M.get("optimizer", {}), M.get("notes", {})
    rob = M.get("uplift_robustness", {})
    SYN = "Trained and evaluated on SYNTHETIC data only; numbers will not transfer to production."
    f = lambda x, d=3: "n/a" if x is None else f"{x:.{d}f}"
    ol = None
    try:
        from ..olist import analyze
        ol = json.loads(analyze.OUT.read_text(encoding="utf8")) if analyze.OUT.exists() else None
    except Exception:
        ol = None
    real = []
    if ol:
        F, T, R = ol["silent_failure"], ol["text_model"], ol["repeat_effect"]
        real = [
            dict(name="REAL DATA: silent-failure model (Olist)", purpose="Predict a bad review although the order arrived on time (a failure delivery telemetry cannot see).",
                 features="price, freight, promise length, seller prior bad-rate (time-lagged), category, state, payment, items, weight, photos.",
                 fallback="Same GBM + isotonic code path as the synthetic model.", status="ACTIVE (real data)",
                 metrics={"AUC (temporal test)": f(F["auc"]), "Top-10% lift": f(F["lift_top10pct"], 2), "Top-10% precision": f(F["top10pct_precision"], 2), "Base rate": f(F["base_rate_test"])},
                 limits=["REAL but weak signal; calibration drifts over time.", "A bad review is a proxy for failure, not a verified failure.", "Olist is a marketplace, not a first-party retailer."]),
            dict(name="REAL DATA: review-text model (Portuguese)", purpose="Predict unhappy customers from free text and surface complaint themes.",
                 features="TF-IDF 1-2 grams of the review comment.", fallback="n/a", status="ACTIVE (real data)",
                 metrics={"AUC (temporal test)": f(T["auc_bad_review_from_text"]), "Late low-reviews that mention delay": f(T["share_of_late_low_reviews_mentioning_delay"], 2)},
                 limits=["Text sentiment vs score is an easy task; this is not a failure-type classifier (no labels exist).", "Theme shares come from a keyword lexicon."]),
            dict(name="REAL DATA: repeat-purchase effect of lateness", purpose="Test the core retention assumption on real customers.",
                 features="Confounders known at purchase (price, freight, promise length, category, state, payment, month).", fallback="n/a", status="ACTIVE (real data)",
                 metrics={"Adjusted effect (pp)": f(R["adjusted_effect_pp"], 2), "95% CI (pp)": f"{R['ci95_pp'][0]:.2f} to {R['ci95_pp'][1]:.2f}", "Significant": R["significant"]},
                 limits=["Observational, not randomised.", "Only about 3% of Olist customers ever reorder, so the outcome is rare.", "Says nothing about the effect of PAYING a credit."])]
    return real + [
        dict(name="LLM incident extraction (primary NLP path)", purpose="Turn one raw complaint into strict JSON {failure_type, incident_key, severity 1-5, customer_noticed}.",
             features="Raw complaint text only.", fallback="Pydantic-validated; on bad JSON / no key -> TF-IDF + logistic regression + regex key.",
             status=("ACTIVE" if llm.available() else "OFFLINE (fallback live)"),
             metrics={"LLM macro-F1": ("n/a (not run offline)" if not nlp.get("llm", {}).get("macro_f1") else f(nlp["llm"]["macro_f1"])),
                      "Baseline macro-F1": f(nlp.get("baseline_macro_f1")), "Baseline accuracy": f(nlp.get("baseline_accuracy"))},
             limits=[SYN, "Template-generated complaints (with mixed wording and hard negatives) are still easier than real tickets.",
                     f"Tickets without an order reference are linked by a recency rule: {f(nlp.get('order_link_inference', {}).get('accuracy'), 2)} accurate on {nlp.get('order_link_inference', {}).get('n_inferred')} tickets.", "LLM path needs OVL_LLM=1 + ANTHROPIC_API_KEY; untested without a key beyond mocked unit tests."]),
        dict(name="Embedding incident discovery (DBSCAN)", purpose="Discover NEW incidents (no predefined label) by clustering complaint embeddings + date, then credit every affected order.",
             features="Complaint embedding (TF-IDF+SVD default, MiniLM optional), order date, warehouse/carrier of the linked order.", fallback="TF-IDF + SVD when sentence-transformers is not configured.",
             status="ACTIVE", metrics={"Planted incidents": inc.get("planted"), "Discovered": len(inc.get("discovered", [])),
                                       **{d["id"] + " order P/R": f"{f(d['order_precision'], 2)} / {f(d['order_recall'], 2)}" for d in inc.get("discovered", [])}},
             limits=[SYN, "Complaint-to-order link is given in the data; in production it must be inferred.", "Statistical test assumes a stable base complaint rate."]),
        dict(name="Silent-failure detector (gradient boosting + isotonic calibration)", purpose="Find likely-damaged orders that have no explicit telemetry, from return/contact patterns.",
             features="delay, amount, tracking checks, returned flag, support contact in 3 days, customer return rate, #orders, category, carrier.",
             fallback="None needed; if the model file is absent the pipeline retrains in ~1 s.", status="ACTIVE",
             metrics={"AUC (test)": f(sf.get("auc")), "Brier raw -> calibrated": f"{f(sf.get('brier_raw'), 4)} -> {f(sf.get('brier_calibrated'), 4)}",
                      "Precision / recall @0.5": f"{f(sf.get('precision_at_0_5'), 2)} / {f(sf.get('recall_at_0_5'), 2)}", "Base rate": f(sf.get("base_rate"))},
             limits=[SYN, "The 'returned' flag dominates; recall is modest at the 0.5 threshold.", "Explanations are occlusion-based, not exact SHAP."]),
        dict(name="Points survival + reward model", purpose="Predict who will NOT redeem before expiry (convert only above a probability threshold) and which reward they would pick.",
             features="Horizon, balance, #redemptions, days since last redemption, prior-cycle redemption rate, tenure, segment; reward shares.",
             fallback="If no probability is available the contract converts every eligible member.", status="ACTIVE",
             metrics={"Survival AUC (within horizon)": f(pts.get("survival", {}).get("auc_within_horizon")), "AUC vs true redemption": f(pts.get("auc_vs_truth")),
                      "Unwanted conversions: convert-all -> thresholded": f"{f(pts.get('convert_all_unwanted_rate'), 3)} -> {f(pts.get('thresholded_unwanted_rate'), 3)}",
                      "Reward accuracy vs naive most-frequent": f"{f(pts.get('reward', {}).get('accuracy'), 3)} vs {f(pts.get('reward', {}).get('baseline_most_frequent_accuracy'), 3)}"},
             limits=[SYN, "Discrete-time survival via classifier, not a Cox/Weibull fit.", "Reward model is NOT better than the naive most-frequent baseline here."]),
        dict(name="Uplift T-learner (two gradient-boosting models)", purpose="Rank customers by predicted RESPONSE to a credit (not churn risk) for goodwill top-ups.",
             features="Segment, tenure, #orders, avg order value, #late, #contacts, exposure $, share noticed.", fallback="Calibration slope shrinks predictions toward 0 (0 => no top-ups).",
             status="ACTIVE", metrics={"Qini coef (uplift)": f(up.get("qini_coef_uplift"), 2), "Qini coef (churn-risk ranking)": f(up.get("qini_coef_risk"), 2),
                                       "Learner chosen": up.get("learner"), "Conservative calibration slope (used)": f(up.get("calibration_slope"), 2),
                                       "Beats churn-risk across 8 pilot draws": f"{rob.get('uplift_beats_risk')} of {rob.get('of')}", "Median slope over draws": f(rob.get("median_slope"), 2), "Pilot size": up.get("n")},
             limits=[SYN, "UNSTABLE: ranks better than churn-risk in only about 3 of 8 repeated pilots; the lower-confidence-bound slope shrinks top-ups to ~0 in most draws.", "Outcomes come from an ASSUMED response function."]),
        dict(name="Constrained settlement optimizer", purpose="Allocate optional top-ups to maximise predicted incremental margin under a budget, with a fairness gate.",
             features="ONLY owed $, uplift estimate, spend estimate (allow-list enforced).", fallback="Zero top-ups when uplift is uncalibrated.", status="ACTIVE",
             metrics={"Fairness gate": ("PASS" if op.get("fairness", {}).get("passed") else "FAIL"), "Max group residual ($)": f(op.get("fairness", {}).get("max_abs_residual_usd"), 4),
                      "Configured budget": op.get("configured_budget")},
             limits=[SYN, "Dose-response curve g(c)=1-exp(-c/4) is an assumption.", "Only checks the attributes present in the data; real deployments need a fuller fairness review."]),
        dict(name="Abuse score (graph ring + IsolationForest + velocity)", purpose="Route suspected rings and anomalies to manual review; never auto-deny.",
             features="Shared device/address/payment graph; claims in 30d, payout in 30d, account age.", fallback="Velocity caps alone.", status="ACTIVE",
             metrics={"Combined P/R": f"{f(ab.get('combined', {}).get('precision'), 2)} / {f(ab.get('combined', {}).get('recall'), 2)}",
                      "Graph-only P/R": f"{f(ab.get('graph_only', {}).get('precision'), 2)} / {f(ab.get('graph_only', {}).get('recall'), 2)}",
                      "Recall: rings / lone heavy / stealth mild": " / ".join(f(ab.get("by_group", {}).get(k, {}).get("combined_recall"), 2) for k in ("coordinated_rings", "lone_heavy_abusers", "stealth_mild_abusers")),
                      "Anomaly-only P/R": f"{f(ab.get('anomaly_only', {}).get('precision'), 2)} / {f(ab.get('anomaly_only', {}).get('recall'), 2)}",
                      "False-positive payout rate": f(ab.get("false_positive_payout_rate"), 3)},
             limits=[SYN, "Mild stealth abusers (nothing shared, low claim rate) are mostly MISSED: low recall by design of the test.", "Anomaly detection alone is too noisy (see precision); it only amplifies graph evidence."]),
        dict(name="LLM-phrased ledger notes + guardrail", purpose="Optionally rephrase notes; a validator rejects apology/promise language or altered amounts and falls back to the template.",
             features="Template note text.", fallback="Deterministic template (always used offline).", status=("ACTIVE" if llm.available() else "OFFLINE (templates)"),
             metrics={"LLM attempts": nt.get("llm_attempts"), "Rejection rate": ("n/a" if nt.get("rejection_rate") is None else f(nt["rejection_rate"], 3)), "Note sources": nt.get("sources")},
             limits=["Regex guardrail cannot catch every possible tone problem.", "Rejection rate is only measurable with the LLM enabled."]),
    ]

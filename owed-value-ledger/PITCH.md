# Kept: pitch

*Every loyalty program keeps a ledger of what customers owe the brand. We built the other half: what the brand owes the customer, paid before they ask.*

## Problem
Customers lose value they were owed (late orders, price drops, expiring or devalued points, paid-but-unused add-ons) and must spend effort to recover it. Effort is the strongest known driver of disloyalty. Fewer than a third of US loyalty members feel valued (Bond 2026), and incumbents optimise the other side: points, tiers, churn scores. Figures are from public industry reports, vendor-sourced in part; not re-verified here.

## Evidence on real data (Olist, ~100k real orders)
- 8.1% of orders arrive late; mean review score falls from 4.3 (on time) to 1.7 (more than a week late).
- **47% of late customers never leave a low review**: claim-based handling misses them. Auto-crediting all late orders costs 0.5% of sales.
- Honest negative: a late first order did **not** measurably reduce repeat purchase (-0.25 pp, CI -0.85 to +0.42), and retention would cover only about 1% of the credit cost. We therefore sell the ledger on trust, fairness and reaching silent victims, not on repeat-purchase margin.
- Our spike detector rediscovered Black Friday 2017 and the May 2018 delivery crisis from the data alone.

- **Real grocery data (Dunnhumby)**: blanket 14-day price protection would cost 1.5% of sales. Targeted-coupon response is positive when matched on past spend (+$6 net per household) but flips negative unmatched: fragile evidence.
- **Real support data (Twitter, 310k complaints)**: 18% of complaints need 3+ customer messages; volume-spike discovery found real outages and delays with no labels.

## Solution
Encode each brand promise as a machine-checkable **promise contract**. Compare promise with reality per customer. Write an IOU into a **customer-visible ledger** and settle it automatically.
- **Settle in value, never in words**: validator-enforced factual notes, no apologies or promises.
- **Salience-aware**: explicit credit when the customer noticed; silent credit when they did not; the experimental subscription contract never sends an "overpaying" alert.
- **Measured, not assumed**: a randomized holdout and uplift-based sizing, not churn risk.

## Novelty (honest)
Pieces exist in silos: Amazon's proactive refunds, UK rail automatic Delay Repay, Netflix cancelling inactive accounts. We do **not** claim first automatic refunds. What is new is the **unified, customer-visible, salience-aware, uplift-measured ledger of the brand's obligations**, with incident discovery that credits silent victims and a single abuse score over graph and anomaly signals.

## Architecture
Rules engine (5 YAML contracts) + calibrated silent-failure GBM + embedding incident discovery + points survival model + uplift T-learner + constrained optimizer + abuse score + guardrailed notes -> idempotent ledger -> FastAPI -> customer and ops views. Runs on CPU in ~40 s total, offline, one command.

## Metrics (synthetic data, assumed response function)
| | |
|---|---|
| Auto-settlement coverage | 87% (rest held for review) |
| Median time-to-settle | 14 h |
| Incidents found | 2 of 2; 159 and 62 silent victims credited |
| Abuse guard (combined) | precision 1.00, recall 0.79: rings 100%, lone heavy 80%, **stealth 0%** |
| Salience guardrail | notify-everything roughly halves the retention gain |
| Net margin | **about break-even (+$0.08 / treated customer)** on assumed response; late-delivery and reward true-ups carry it, five other contracts are net-negative |
| Uplift sizing | unstable (3 of 8 repeated pilots beat churn-risk); top-ups stay near 0 until a bigger pilot |
| Holdout power | ~6,200 customers per arm to see a 2 pp effect |

The finding is useful: automation is not free, and the per-contract table says which promises to settle automatically (`--wedge`), which to cap and which to redesign.

## Risks
Abuse and gaming (graph + anomaly + velocity, human review, never auto-deny); salience backfire (policy + holdout guardrail); finance resistance to converting breakage; thresholds gamed operationally (graduated bands); regulation if amounts vary with personal data (input allow-list and residual audit); LLM tone (validator, template fallback).

## Ask / next step
Run a live 10% randomized holdout on one contract (late delivery) for 90 days. **All results above come from synthetic data; real impact needs that holdout.**

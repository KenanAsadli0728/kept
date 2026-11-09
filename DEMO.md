# 3-minute demo script

Start: `python run.py`, open http://127.0.0.1:8000. Say once, up front: *"All data is synthetic."*

## 0:00 - The hook (20 s)
"Every loyalty program in this room keeps a ledger of what customers owe the brand. We built the other half: what the brand owes the customer, paid before they ask."

## 0:20 - Maya, Customer view (75 s)
The page opens on **Maya R.**: *"We owe you $49.39."* Walk down the list:
1. **Late delivery** (47 min): an *explicit* credit with one factual line, "arrived 47 min after the promised time. $3.00 credited". "She waited and contacted support, so she is told. No apology, no promise: value, not words."
2. **Price protection** ($20 headphone drop) appears *quietly*. "She never noticed. Telling her would only make her look at prices."
3. **Points**: 1,800 points expiring Friday: 400 became 2 Iced lattes (her usual reward, capped at $10) and the other 1,400 were *extended 90 days* rather than voided. Open **Why this entry**: the survival model gave a 90% chance she would not redeem on her own.
4. **Unused streaming add-on** paused with a prorated credit. "Experimental, silent, and it never says 'you're overpaying', which can trigger churn."
5. **Reward true-up**: Iced latte went from 150 to 200 points, so we restored 600 points.

"Four problems, zero contacts, zero claims." Click **Simulate new event** twice: a late order lands (explicit), then a price drop (silent), live.

## 1:35 - Flip to Ops (80 s)
- **Overview**: value settled, coverage 87%, median 14 h.
- **Incidents**: "One warehouse failure. 167 customers affected, 8 complained, 159 silent victims credited automatically. Found by clustering complaint text and dates, then statistically tested so we don't invent incidents."
- **Abuse graph**: the red cluster is a ring sharing devices and payment cards. Their payouts sit in the **Review queue**; we approve or deny, and nothing is auto-denied.
- **Experiment**: the holdout, Qini curve, salience guardrail ("apologise for everything" halves the gain), and the by-contract economics.

## 2:30 - Real data (25 s)
Ops -> **Real data (Olist)**: "This is real: 100k Brazilian orders. 8% late, and 47% of those customers never complained: a claim form would miss half. Our detector found Black Friday 2017 on its own. And we tested our own thesis: lateness did not measurably cut repeat purchases, so we sell this on trust and fairness, not retention margin."

## 2:55 - The honest close (25 s)
"On synthetic data with an assumed response, the ledger is about break-even, and two contracts, late delivery and reward true-ups, carry it. The others cost more than the retention they buy, and we say so. We also show where we fail: stealth abusers slip through. A randomized holdout on a live pilot is how this gets settled."

## If asked
- *Is the LLM used?* Optional. Offline, a TF-IDF fallback runs; LLM paths are tested with mocked responses.
- *Why is the uplift model not spending?* It is unstable across repeated pilots (better than churn-risk in 3 of 8), so a conservative calibration slope keeps top-ups near zero until a larger live pilot shows real heterogeneity. The Experiment and Models tabs show this.

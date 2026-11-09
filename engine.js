/* Owed-Value Ledger — core engine (no DOM). Works in browser (window.OVL) and Node (module.exports). */
(function (root) {
  'use strict';

  // ---------- seeded RNG ----------
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------- default promise-contract parameters ----------
  const DEFAULT_PARAMS = {
    lateMin: 15,        // grace before a delivery counts as late
    latePct: 10,        // % of order value at full severity
    lateCap: 5,         // $ cap per late order
    lateRamp: 45,       // minutes past grace to reach full severity (graduated, no cliff)
    priceWindow: 14,    // days of price protection
    expDays: 7,         // points considered "about to expire"
    minPts: 500,        // minimum balance worth auto-converting
    ptValue: 0.005,     // $ per point
    unusedDays: 60,     // days without usage before add-on is flagged
    velocity: 4,        // max auto-settled late credits per customer / 30d
    ringRatio: 2.5      // cluster avg owed vs global avg to flag
  };

  const TYPES = {
    late:     { label: 'Late delivery',        salience: 'explicit', icon: '⏱' },
    price:    { label: 'Price-protection',     salience: 'silent',   icon: '↓$' },
    points:   { label: 'Expiring points',      salience: 'silent',   icon: '★' },
    sub:      { label: 'Unused subscription',  salience: 'silent',   icon: '⏸' },
    incident: { label: 'Damaged-batch incident', salience: 'explicit', icon: '⚠' }
  };

  const REWARDS = ['Iced latte', 'Free delivery pass', 'Pastry of the day', '$5 basket credit', 'Free dessert'];
  const NAMES = ['Ava','Liam','Noor','Mateo','Yuki','Sofia','Omar','Chen','Zara','Ivan','Lena','Diego','Amira','Kofi','Priya','Elif','Hugo','Mina','Tariq','Nora'];

  // ---------- synthetic data ----------
  function generate(n, seed) {
    const r = rng(seed || 7);
    const customers = [];
    const ri = (a, b) => a + Math.floor(r() * (b - a + 1));
    const money = (a, b) => Math.round((a + r() * (b - a)) * 100) / 100;

    for (let i = 1; i <= n; i++) {
      const c = {
        id: i,
        name: NAMES[ri(0, NAMES.length - 1)] + ' ' + String.fromCharCode(65 + ri(0, 25)) + '.',
        device: 'D' + i, addr: 'A' + i, pay: 'P' + i,
        ring: false, orders: [], sub: null, points: null
      };
      // households share an address (benign graph noise)
      if (r() < 0.06 && i > 1) c.addr = 'A' + ri(1, i - 1);
      const k = ri(2, 10);
      for (let j = 0; j < k; j++) {
        const o = {
          id: i + '-' + j, ageDays: ri(0, 364), total: money(12, 140),
          lateMin: r() < 0.08 ? ri(5, 90) : 0, wh: ri(0, 3),
          dropAmt: r() < 0.06 ? money(3, 30) : 0, dropAfter: ri(1, 10)
        };
        c.orders.push(o);
      }
      if (r() < 0.45) c.sub = { fee: money(4.99, 19.99), daysUnused: r() < 0.3 ? ri(45, 150) : ri(0, 30), name: 'Add-on' };
      const bal = Math.round(Math.pow(r(), 1.8) * 6000);
      c.points = { bal, daysToExpiry: r() < 0.04 ? ri(1, 7) : ri(8, 400), reward: REWARDS[ri(0, REWARDS.length - 1)] };
      customers.push(c);
    }

    // inject abuse rings: shared device/payment, inflated late claims
    const ringCount = Math.max(2, Math.round(n / 800));
    for (let g = 0; g < ringCount; g++) {
      const size = ri(6, 10);
      const dev = 'DR' + g, pay = 'PR' + g;
      for (let m = 0; m < size; m++) {
        const c = customers[ri(0, n - 1)];
        if (c.ring) continue;
        c.ring = true; c.device = dev; if (m % 2) c.pay = pay;
        const lateCount = m < size / 2 ? ri(5, 8) : ri(2, 3); // some are low-velocity, only graph catches them
        for (let q = 0; q < lateCount; q++)
          c.orders.push({ id: c.id + '-r' + q, ageDays: ri(0, 25), total: money(40, 90), lateMin: ri(40, 120), wh: ri(0, 3), dropAmt: 0, dropAfter: 1 });
      }
    }

    // the hero customer
    customers[0] = makeMaya();
    return customers;
  }

  function makeMaya() {
    return {
      id: 0, name: 'Maya R.', device: 'D-maya', addr: 'A-maya', pay: 'P-maya', ring: false,
      orders: [
        { id: 'M-1', label: 'Weekly groceries', ageDays: 2, total: 62.4, lateMin: 47, wh: 1, dropAmt: 0, dropAfter: 1 },
        { id: 'M-2', label: 'Wireless headphones', ageDays: 4, total: 129, lateMin: 0, wh: 2, dropAmt: 20, dropAfter: 3 },
        { id: 'M-3', label: 'Pantry restock', ageDays: 40, total: 38.2, lateMin: 0, wh: 0, dropAmt: 0, dropAfter: 1 }
      ],
      sub: { fee: 9.99, daysUnused: 70, name: 'Streaming add-on' },
      points: { bal: 1800, daysToExpiry: 3, reward: 'Iced latte' }
    };
  }

  // ---------- promise contracts ----------
  // Each contract: customer + params -> list of IOUs. Deterministic, machine-checkable.
  const CONTRACTS = [
    {
      id: 'late_delivery', type: 'late',
      yaml: (P) => `late_delivery:\n  when: actual - promised > ${P.lateMin}m   # graduated, no cliff\n  credit: min($${P.lateCap}, ${P.latePct}% of order * min(1, over/${P.lateRamp}m))\n  salience: explicit   # customer waited -> short factual note, no promises`,
      run(c, P) {
        const out = [];
        for (const o of c.orders) {
          if (o.ageDays > 30 || !o.lateMin) continue;
          const over = o.lateMin - P.lateMin;
          if (over <= 0) continue;
          const amt = Math.min(P.lateCap, o.total * (P.latePct / 100) * Math.min(1, over / P.lateRamp));
          if (amt < 0.25) continue;
          out.push({ type: 'late', amount: amt, ref: o.id, label: o.label || 'Order ' + o.id,
            note: `Arrived ${o.lateMin} min after the promised time (${over} min past grace).`,
            meta: { lateMin: o.lateMin } });
        }
        return out;
      }
    },
    {
      id: 'price_protection', type: 'price',
      yaml: (P) => `price_protection:\n  when: price_drop within ${P.priceWindow}d of purchase\n  credit: price_paid - lowest_price_in_window\n  salience: silent     # nothing to apologise for; just appears`,
      run(c, P) {
        const out = [];
        for (const o of c.orders) {
          if (!o.dropAmt || o.ageDays > P.priceWindow || o.dropAfter > o.ageDays || o.dropAfter > P.priceWindow) continue;
          out.push({ type: 'price', amount: o.dropAmt, ref: o.id, label: o.label || 'Order ' + o.id,
            note: `Price dropped $${o.dropAmt.toFixed(2)} ${o.dropAfter} day(s) after you bought it. Difference refunded.` });
        }
        return out;
      }
    },
    {
      id: 'expiring_points', type: 'points',
      yaml: (P) => `expiring_points:\n  when: expiry <= ${P.expDays}d and balance >= ${P.minPts}\n  action: auto-convert to member's most-used reward ($${P.ptValue}/pt)\n  salience: silent`,
      run(c, P) {
        const p = c.points;
        if (!p || p.daysToExpiry > P.expDays || p.bal < P.minPts) return [];
        return [{ type: 'points', amount: p.bal * P.ptValue, ref: 'points', label: p.bal.toLocaleString() + ' points',
          note: `Would have expired in ${p.daysToExpiry} day(s). Converted to your usual reward: ${p.reward}.` }];
      }
    },
    {
      id: 'unused_subscription', type: 'sub',
      yaml: (P) => `unused_subscription:\n  when: 0 usage for >= ${P.unusedDays}d\n  action: pause + prorated credit (framed as value-preserving)\n  credit: fee * min(1, days_unused / 90)\n  salience: silent + opt-out   # testing vs holdout: overpayment alerts can trigger exit`,
      run(c, P) {
        const s = c.sub;
        if (!s || s.daysUnused < P.unusedDays) return [];
        return [{ type: 'sub', amount: s.fee * Math.min(1, s.daysUnused / 90), ref: 'sub', label: s.name,
          note: `Unopened for ${s.daysUnused} days. Paused, with a prorated credit. Resume any time.` }];
      }
    },
    {
      id: 'incident', type: 'incident',
      yaml: () => `incident:\n  when: ops declares bad batch (warehouse, last 7d) -> IOU for EVERYONE hit, not just complainers\n  credit: min($15, 25% of order)\n  salience: explicit`,
      run(c, P) {
        const out = [];
        if (P.incidentWh == null) return out;
        for (const o of c.orders) {
          if (o.wh !== P.incidentWh || o.ageDays > 7) continue;
          out.push({ type: 'incident', amount: Math.min(15, o.total * 0.25), ref: o.id, label: o.label || 'Order ' + o.id,
            note: `Part of a batch from warehouse ${'ABCD'[o.wh]} we found was handled badly. Credited without you asking.` });
        }
        return out;
      }
    }
  ];

  // ---------- engine + abuse guard ----------
  function evaluate(c, P) {
    const ious = [];
    for (const k of CONTRACTS) for (const i of k.run(c, P)) {
      i.contract = k.id; i.salience = TYPES[i.type].salience; i.amount = Math.round(i.amount * 100) / 100;
      i.status = 'settled'; ious.push(i);
    }
    return ious;
  }

  function components(customers) {
    const parent = new Map();
    const find = (x) => { while (parent.get(x) !== x) { parent.set(x, parent.get(parent.get(x))); x = parent.get(x); } return x; };
    const add = (x) => { if (!parent.has(x)) parent.set(x, x); };
    const union = (a, b) => { add(a); add(b); const ra = find(a), rb = find(b); if (ra !== rb) parent.set(ra, rb); };
    for (const c of customers) { const k = 'c' + c.id; add(k); union(k, 'd' + c.device); union(k, 'a' + c.addr); union(k, 'p' + c.pay); }
    const groups = new Map();
    for (const c of customers) {
      const g = find('c' + c.id);
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(c);
    }
    return groups;
  }

  function run(customers, P) {
    const ledger = new Map();
    let total = 0, n = 0;
    for (const c of customers) {
      const ious = evaluate(c, P);
      ledger.set(c.id, ious);
      for (const i of ious) { total += i.amount; n++; }
    }
    const globalAvg = total / Math.max(1, customers.length);

    // abuse guard 1: velocity on late credits
    for (const c of customers) {
      const L = ledger.get(c.id).filter(i => i.type === 'late');
      if (L.length > P.velocity) for (const i of L) { i.status = 'review'; i.reason = 'velocity'; }
    }
    // abuse guard 2: connected components over shared device/address/payment
    const flagged = [];
    for (const [, members] of components(customers)) {
      if (members.length < 3) continue;
      const sum = members.reduce((s, c) => s + ledger.get(c.id).reduce((a, i) => a + i.amount, 0), 0);
      const avg = sum / members.length;
      if (avg > globalAvg * P.ringRatio) {
        let held = 0;
        for (const c of members) for (const i of ledger.get(c.id)) {
          if (i.type === 'late' || i.type === 'incident') { if (i.status !== 'review') { i.status = 'review'; i.reason = 'ring'; } held += i.amount; }
        }
        flagged.push({ size: members.length, avg, held, truthRing: members.filter(c => c.ring).length, members: members.map(c => c.id) });
      }
    }
    flagged.sort((a, b) => b.held - a.held);

    // stats
    const byType = {}; let settled = 0, held = 0, withIOU = 0, nSettled = 0, nHeld = 0;
    for (const c of customers) {
      const L = ledger.get(c.id); if (L.length) withIOU++;
      for (const i of L) {
        byType[i.type] = byType[i.type] || { n: 0, amt: 0 };
        if (i.status === 'settled') { settled += i.amount; nSettled++; byType[i.type].n++; byType[i.type].amt += i.amount; }
        else { held += i.amount; nHeld++; }
      }
    }
    // abuse precision/recall vs ground truth (synthetic only)
    let tp = 0, fp = 0, fn = 0;
    const heldIds = new Set();
    for (const c of customers) if (ledger.get(c.id).some(i => i.status === 'review')) heldIds.add(c.id);
    for (const c of customers) {
      const h = heldIds.has(c.id);
      if (h && c.ring) tp++; else if (h && !c.ring) fp++; else if (!h && c.ring) fn++;
    }
    return { ledger, stats: { total, nIOUs: n, settled, held, nSettled, nHeld, withIOU, byType, flagged,
      abuse: { tp, fp, fn, precision: tp / Math.max(1, tp + fp), recall: tp / Math.max(1, tp + fn) } } };
  }

  // ---------- holdout experiment (ASSUMPTION-DRIVEN simulation) ----------
  const DEFAULT_SIM = {
    holdoutPct: 10, base: 0.78, hit: 0.04, recover: 0.65, claimedRecover: 0.5, hassle: 0.02,
    moneyEffect: 0.004, moneyCap: 25, silentTrust: 0.004, backfire: 0.01,
    policy: 'salience',   // 'salience' = noticed->explicit, unnoticed->silent;  'notifyAll' = apologise for everything
    margin: 140, contactCost: 6
  };
  const CLAIM_P = { late: 0.25, price: 0.12, points: 0, sub: 0.05, incident: 0.15 };
  const NOTICED = { late: true, incident: true, price: false, points: false, sub: false };

  function simulate(customers, ledger, S, seed) {
    const r = rng(seed || 99);
    const arms = { T: { n: 0, p: 0, v: 0, pay: 0, contacts: 0 }, H: { n: 0, p: 0, v: 0, pay: 0, contacts: 0 } };
    for (const c of customers) {
      const L = ledger.get(c.id).filter(i => i.status === 'settled');
      if (!L.length) continue;
      const arm = r() < S.holdoutPct / 100 ? 'H' : 'T';
      let p = S.base, pay = 0, contacts = 0;
      if (arm === 'T') {
        let money = 0;
        for (const i of L) {
          money += i.amount; pay += i.amount;
          if (NOTICED[i.type]) p -= S.hit * (1 - S.recover);
          else {
            p += S.silentTrust;
            if (S.policy === 'notifyAll') p -= S.backfire;   // apologising for failures the customer never noticed
          }
        }
        p += S.moneyEffect * Math.min(S.moneyCap, money);
      } else {
        let money = 0;
        for (const i of L) {
          const cp = CLAIM_P[i.type]; contacts += cp;
          if (NOTICED[i.type]) p -= S.hit * (1 - cp * S.claimedRecover) + S.hassle * cp;
          money += cp * i.amount; pay += cp * i.amount;
        }
        p += S.moneyEffect * Math.min(S.moneyCap, money);
      }
      p = Math.max(0, Math.min(1, p));
      const a = arms[arm]; a.n++; a.p += p; a.v += p * (1 - p); a.pay += pay; a.contacts += contacts;
    }
    const T = arms.T, H = arms.H;
    const pT = T.p / Math.max(1, T.n), pH = H.p / Math.max(1, H.n);
    const se = Math.sqrt(T.v / Math.max(1, T.n) / Math.max(1, T.n) + H.v / Math.max(1, H.n) / Math.max(1, H.n));
    const uplift = pT - pH;
    const payT = T.pay / Math.max(1, T.n), payH = H.pay / Math.max(1, H.n);
    const ctH = H.contacts / Math.max(1, H.n);
    const perCustomer = uplift * S.margin - (payT - payH) + ctH * S.contactCost;
    return { nT: T.n, nH: H.n, pT, pH, uplift, se, payT, payH, contactsAvoided: ctH * T.n,
      perCustomer, net: perCustomer * T.n, retainedExtra: uplift * T.n };
  }

  const API = { rng, DEFAULT_PARAMS, DEFAULT_SIM, TYPES, CONTRACTS, generate, makeMaya, evaluate, run, simulate };
  if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.OVL = API;
})(typeof window !== 'undefined' ? window : globalThis);

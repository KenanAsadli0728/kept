import json
from sqlalchemy import create_engine, text
from . import config

_engine = None


def get_engine(path=None):
    global _engine
    if path is not None:
        return create_engine(f"sqlite:///{path}", future=True)
    if _engine is None:
        _engine = create_engine(f"sqlite:///{config.DB_PATH}", future=True)
    return _engine


DDL = """
DROP TABLE IF EXISTS ledger_entries;
DROP TABLE IF EXISTS abuse_flags;
DROP TABLE IF EXISTS promise_contracts;
DROP TABLE IF EXISTS points_adjustments;
CREATE TABLE points_adjustments(id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id INTEGER, used_points INTEGER, extend_days INTEGER, ledger_ref TEXT UNIQUE);
CREATE TABLE promise_contracts(id TEXT PRIMARY KEY, type TEXT, params_json TEXT, active INTEGER, experimental INTEGER);
CREATE TABLE ledger_entries(
  id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id INTEGER, contract_type TEXT, source_ref TEXT,
  amount REAL, currency_or_points TEXT, value_usd REAL, status TEXT, noticed_flag INTEGER,
  settlement_mode TEXT, note TEXT, reasons TEXT, note_source TEXT, event_at TEXT, created_at TEXT,
  settled_at TEXT, experiment_arm TEXT, review_reason TEXT, uplift_score REAL, topup REAL,
  UNIQUE(contract_type, source_ref));
CREATE INDEX ix_ledger_cust ON ledger_entries(customer_id);
CREATE TABLE abuse_flags(id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id INTEGER, reason TEXT, score REAL, created_at TEXT);
"""


def init_runtime_tables(eng):
    with eng.begin() as c:
        for stmt in [s for s in DDL.split(";") if s.strip()]:
            c.execute(text(stmt))


def load_contract_rows(eng):
    with eng.connect() as c:
        rows = c.execute(text("SELECT id,type,params_json,active,experimental FROM promise_contracts")).fetchall()
    return {r[0]: dict(id=r[0], type=r[1], params=json.loads(r[2]), active=bool(r[3]), experimental=bool(r[4])) for r in rows}

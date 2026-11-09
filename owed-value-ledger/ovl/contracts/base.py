from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class IOU:
    """One machine-checked broken promise. `note` is a short factual template (never apology/promise language)."""
    contract_type: str
    source_ref: str                 # unique per (contract_type, source_ref) -> idempotency key
    amount: float                   # in `unit`
    unit: str                       # 'USD' | 'POINTS'
    value_usd: float
    event_at: datetime
    note: str
    meta: dict = field(default_factory=dict)
    customer_id: int = 0


def money(x):
    return round(float(x) + 1e-9, 2)

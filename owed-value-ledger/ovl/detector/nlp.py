"""Complaint understanding.

Primary path: LLM -> strict JSON -> pydantic validation.
Fallback (no key, API error, invalid JSON): TF-IDF + logistic regression + regex incident key.
"""
import json
import re
from typing import Literal, Optional

import joblib
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .. import config, llm
from .texts import TYPES

FailureType = Literal["late", "damaged", "spoiled", "not_received", "price", "refund_status", "other"]


class IncidentKey(BaseModel):
    carrier: Optional[str] = None
    warehouse: Optional[str] = None
    date: Optional[str] = None


class Extraction(BaseModel):
    failure_type: FailureType
    incident_key: Optional[IncidentKey] = None
    severity: int = Field(ge=1, le=5)
    customer_noticed: bool

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce(cls, v):
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v


SYSTEM = ("You extract structured facts from one customer complaint. Reply with ONLY a JSON object: "
          '{"failure_type": one of ' + json.dumps(TYPES) + ', "incident_key": {"carrier": str|null, "warehouse": str|null, "date": str|null} or null, '
          '"severity": integer 1-5, "customer_noticed": boolean}. No prose.')

MODEL_PATH = config.MODELS_DIR / "nlp_baseline.joblib"


def train_baseline(texts, labels):
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True), LogisticRegression(max_iter=1000, C=5))
    pipe.fit(texts, labels)
    return pipe


def load_or_train(eng):
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    df = pd.read_sql("SELECT text,label FROM nlp_training", eng)
    pipe = train_baseline(df.text, df.label)
    joblib.dump(pipe, MODEL_PATH)
    return pipe


_WH = re.compile(r"warehouse\s+(W\d)", re.I)
_CAR = re.compile(r"carrier\s+([A-C])\b", re.I)
_DATE = re.compile(r"on\s+([A-Z][a-z]{2}\s+\d{1,2})")
_SEV = [(re.compile(r"spoil|smashed|broken|crushed|never receiv|nothing arrived", re.I), 4),
        (re.compile(r"late|delayed|waiting|warm|melted", re.I), 3)]


def regex_key(text):
    wh, car, d = _WH.search(text), _CAR.search(text), _DATE.search(text)
    if not (wh or car or d):
        return None
    return IncidentKey(warehouse=wh.group(1).upper() if wh else None, carrier=car.group(1).upper() if car else None, date=d.group(1) if d else None)


def fallback_extract(text, baseline) -> Extraction:
    ft = str(baseline.predict([text])[0])
    sev = next((s for rx, s in _SEV if rx.search(text)), 2)
    return Extraction(failure_type=ft, incident_key=regex_key(text), severity=sev, customer_noticed=ft not in ("price", "other"))


def parse_llm_json(raw) -> Optional[Extraction]:
    """Strict parse + schema validation. Returns None on ANY problem (caller falls back)."""
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return Extraction.model_validate(json.loads(m.group(0)))
    except (ValueError, ValidationError):
        return None


def extract(text, baseline):
    """Returns (Extraction, source) where source is 'llm' or 'fallback'."""
    raw = llm.complete(text, system=SYSTEM, max_tokens=200)
    ex = parse_llm_json(raw)
    if ex is not None:
        return ex, "llm"
    return fallback_extract(text, baseline), "fallback"

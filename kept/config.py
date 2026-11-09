"""Global config. The demo runs on a fixed 'demo clock' so everything is deterministic."""
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("KEPT_DATA_DIR", ROOT / "data"))
DB_PATH = DATA_DIR / "kept.db"
MODELS_DIR = DATA_DIR / "models"
METRICS_PATH = DATA_DIR / "metrics.json"
CONTRACTS_YAML = ROOT / "config" / "contracts.yaml"
STATIC_DIR = ROOT / "kept" / "static"

SEED = 42
N_CUSTOMERS = int(os.environ.get("KEPT_N_CUSTOMERS", 5000))
NOW = datetime(2026, 1, 15, 12, 0, 0)        # demo clock
MAYA_ID = 1
LLM_ENABLED = os.environ.get("KEPT_LLM", "0") == "1" and bool(os.environ.get("ANTHROPIC_API_KEY"))
LLM_MODEL = os.environ.get("KEPT_LLM_MODEL", "claude-haiku-5-5")
EMBED_BACKEND = os.environ.get("KEPT_EMBED", "tfidf")   # 'tfidf' (default) or 'st'

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

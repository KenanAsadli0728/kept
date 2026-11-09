import os
import tempfile

# a small, isolated world for the whole test session (must be set BEFORE importing kept)
_TMP = tempfile.mkdtemp(prefix="kept_test_")
os.environ["KEPT_DATA_DIR"] = _TMP
os.environ["KEPT_N_CUSTOMERS"] = "1500"
os.environ.pop("KEPT_LLM", None)

import warnings
warnings.filterwarnings("ignore")
import pytest


@pytest.fixture(scope="session")
def world():
    from kept import engine, seed
    from kept.eval import training
    eng = seed.build()
    models = training.train_all(eng)
    first = engine.run_pipeline(eng, models=models)
    return dict(eng=eng, models=models, first=first)


@pytest.fixture(scope="session")
def ledger(world):
    import pandas as pd
    return pd.read_sql("SELECT * FROM ledger_entries", world["eng"])

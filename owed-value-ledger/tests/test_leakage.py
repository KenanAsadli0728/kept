"""No model may use `_ground_truth` as a feature. Static + structural checks."""
import ast
from pathlib import Path
import pandas as pd
from ovl.detector import silent_failure as sf, redemption as rd
from ovl.policy import uplift, optimizer

ROOT = Path(__file__).resolve().parent.parent / "ovl"
ALLOWED_FILES = {"seed.py"}            # generates the truth
ALLOWED_DIRS = {"eval"}                # labels / hidden response / metrics (evaluation only)


def _code_strings(path):
    """String literals that are CODE (SQL etc.), excluding docstrings."""
    tree = ast.parse(path.read_text(encoding="utf8"))
    doc = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body                 and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant):
            doc.add(id(n.body[0].value))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc]


def test_ground_truth_only_referenced_by_seed_and_eval():
    offenders = []
    for f in ROOT.rglob("*.py"):
        rel = f.relative_to(ROOT)
        if f.name in ALLOWED_FILES or (rel.parts and rel.parts[0] in ALLOWED_DIRS):
            continue
        if any("_ground_truth" in s for s in _code_strings(f)):
            offenders.append(str(rel))
    assert offenders == [], offenders


def test_feature_lists_exclude_ground_truth_columns(world):
    gt_cols = set(pd.read_sql("SELECT * FROM _ground_truth LIMIT 1", world["eng"]).columns) - {"kind", "ref_id", "label"}
    for feats in (sf.FEATURES, rd.SFEATS, uplift.FEATURES, optimizer.ALLOWED_INPUTS):
        assert not (set(feats) & gt_cols), set(feats) & gt_cols
    assert {"true_response", "base_retention", "backfire", "is_ring", "silent_failure", "will_redeem", "spend_90d"} <= gt_cols


def test_detector_feature_matrix_has_only_declared_columns(world):
    from ovl import data
    t = data.load_tables(world["eng"])
    X = sf.build_features(t["orders"], t["contacts"])
    assert set(X.columns) == set(sf.FEATURES) | {"customer_id"}


def test_fitted_detector_uses_only_declared_features(world):
    assert list(world["models"]["silent"].ref.index) == sf.FEATURES

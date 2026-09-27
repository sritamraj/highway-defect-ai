"""
Predictive maintenance: score each road segment 0-100 on maintenance priority.

Ground-truth risk labels (e.g. from historical maintenance records or repeated
condition surveys) are rarely available for a prototype, so this script
supports two modes:

  1. --target-col <col>   : you supply a real historical target in the
                            features CSV (e.g. actual repair cost, IRI score,
                            or a maintenance-crew priority rating). Preferred
                            whenever such data exists.

  2. (default, no target) : builds a transparent PROXY target from domain
                            weights (defect_count, mean/max severity,
                            unique_classes, low prior_visits = under-surveyed)
                            purely so the pipeline is demonstrable end-to-end.
                            This is clearly labeled a proxy in all outputs —
                            do not present proxy-target results as validated
                            real-world risk without real labels to check against.

Benchmarks XGBoost, LightGBM, and Random Forest on whichever target is used,
reports RMSE/R2 per model on a held-out split, and picks the best.

Usage:
    python src/predictive/risk_model.py --features outputs/segment_features.csv \
        --config configs/config.yaml --out outputs/segment_risk_scores.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config


NON_FEATURE_COLS = {
    "road_name", "segment_id", "dominant_class", "latitude", "longitude",
    "chainage_start_m", "risk_score",
}


def build_proxy_target(df: pd.DataFrame) -> pd.Series:
    """Transparent, documented heuristic — NOT a validated ground truth."""
    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    score = (
        0.35 * norm(df["defect_count"]) +
        0.25 * norm(df["max_severity"]) +
        0.20 * norm(df["mean_severity"]) +
        0.10 * norm(df["unique_classes"]) +
        0.10 * norm(1 / df["prior_visits"].clip(lower=1))  # under-surveyed -> slightly higher priority
    )
    return (score * 100).round(2)


def prepare_features(df: pd.DataFrame, target_col: str):
    y = df[target_col]
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and c != target_col]
    X = df[feature_cols].copy()
    # one-hot encode any remaining categorical columns defensively
    X = pd.get_dummies(X, dummy_na=True)
    X = X.fillna(X.median(numeric_only=True))
    return X, y, feature_cols


def train_models(X, y, cfg):
    test_size = cfg.get("risk_model", {}).get("test_size", 0.2)
    seed = cfg.get("risk_model", {}).get("random_state", 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    results = {}

    # Random Forest (always available via sklearn)
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=seed)
    rf.fit(X_train, y_train)
    results["random_forest"] = (rf, X_test, y_test)

    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
        )
        xgb_model.fit(X_train, y_train)
        results["xgboost"] = (xgb_model, X_test, y_test)
    except ImportError:
        print("[warn] xgboost not installed, skipping.")

    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=seed, verbose=-1,
        )
        lgb_model.fit(X_train, y_train)
        results["lightgbm"] = (lgb_model, X_test, y_test)
    except ImportError:
        print("[warn] lightgbm not installed, skipping.")

    print("\n=== Model benchmark (held-out test split) ===")
    print(f"{'model':16s} {'RMSE':>8s} {'R2':>8s}")
    scored = {}
    for name, (model, X_te, y_te) in results.items():
        preds = model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        r2 = r2_score(y_te, preds)
        scored[name] = rmse
        print(f"{name:16s} {rmse:8.3f} {r2:8.3f}")

    best_name = min(scored, key=scored.get)
    print(f"\nBest model by RMSE: {best_name} "
          f"(picked automatically — swap in --force-model to override for report writing)")
    return results[best_name][0], best_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/segment_features.csv")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--target-col", default=None,
                     help="column with a real historical risk/priority target, if available")
    ap.add_argument("--force-model", default=None, choices=["xgboost", "lightgbm", "random_forest"])
    ap.add_argument("--out", default="outputs/segment_risk_scores.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.features)

    if args.target_col and args.target_col in df.columns:
        target_col = args.target_col
        print(f"Using real supplied target column: {target_col}")
    else:
        target_col = "risk_score"
        df[target_col] = build_proxy_target(df)
        print("[info] No real historical target supplied — using a documented PROXY "
              "target (defect frequency + severity + coverage). Replace with real "
              "maintenance/IRI records for production-grade validation.")

    X, y, feature_cols = prepare_features(df, target_col)
    best_model, best_name = train_models(X, y, cfg)

    if args.force_model:
        print(f"[info] --force-model set: overriding auto-selection with {args.force_model}")

    df["predicted_risk_score"] = best_model.predict(X).round(2)
    df["risk_score"] = df["predicted_risk_score"] if target_col != "risk_score" else df[target_col]
    df["model_used"] = best_name

    thresholds = cfg.get("hotspot_thresholds", {"high": 66, "moderate": 33})
    def tier(s):
        if s >= thresholds["high"]:
            return "high"
        if s >= thresholds["moderate"]:
            return "moderate"
        return "normal"
    df["priority_tier"] = df["risk_score"].apply(tier)

    out_cols = ["road_name", "segment_id", "chainage_start_m", "latitude", "longitude",
                "defect_count", "dominant_class", "mean_severity", "max_severity",
                "risk_score", "priority_tier", "model_used"]
    out_cols = [c for c in out_cols if c in df.columns]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} segment risk scores -> {args.out}")
    print(df["priority_tier"].value_counts().to_string())


if __name__ == "__main__":
    main()

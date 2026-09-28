"""
Predictive maintenance: score each road segment 0-100 on maintenance priority.

Ground-truth risk labels (e.g. from historical maintenance records or repeated
condition surveys) are rarely available for a prototype, so this script
supports two modes:

  1. --target-col <col> : supply a real historical target in the features CSV.

  2. Default: builds a transparent PROXY target from domain weights.

The script benchmarks XGBoost, LightGBM, and Random Forest and either:
  - automatically selects the lowest-RMSE model, or
  - uses the model specified by --force-model.
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
    "road_name",
    "segment_id",
    "dominant_class",
    "latitude",
    "longitude",
    "chainage_start_m",
    "risk_score",
}


def build_proxy_target(df: pd.DataFrame) -> pd.Series:
    """Transparent, documented heuristic — NOT validated ground truth."""

    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    score = (
        0.35 * norm(df["defect_count"])
        + 0.25 * norm(df["max_severity"])
        + 0.20 * norm(df["mean_severity"])
        + 0.10 * norm(df["unique_classes"])
        + 0.10 * norm(
            1 / df["prior_visits"].clip(lower=1)
        )
    )

    return (score * 100).round(2)


def prepare_features(df: pd.DataFrame, target_col: str):
    y = df[target_col]

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and c != target_col
    ]

    X = df[feature_cols].copy()

    # One-hot encode categorical columns.
    X = pd.get_dummies(X, dummy_na=True)

    # Fill missing numeric values.
    X = X.fillna(X.median(numeric_only=True))

    return X, y, feature_cols


def train_models(X, y, cfg):
    test_size = cfg.get("risk_model", {}).get("test_size", 0.2)
    seed = cfg.get("risk_model", {}).get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    models = {}

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=seed,
    )

    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    # ---------------------------------------------------------
    # XGBoost
    # ---------------------------------------------------------
    try:
        import xgboost as xgb

        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
        )

        xgb_model.fit(X_train, y_train)
        models["xgboost"] = xgb_model

    except ImportError:
        print("[warn] xgboost not installed, skipping.")

    # ---------------------------------------------------------
    # LightGBM
    # ---------------------------------------------------------
    try:
        import lightgbm as lgb

        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=seed,
            verbose=-1,
        )

        lgb_model.fit(X_train, y_train)
        models["lightgbm"] = lgb_model

    except ImportError:
        print("[warn] lightgbm not installed, skipping.")

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------
    print("\n=== Model benchmark (held-out test split) ===")
    print(f"{'model':16s} {'RMSE':>8s} {'R2':>8s}")

    benchmark = {}

    for name, model in models.items():
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        benchmark[name] = {
            "rmse": rmse,
            "r2": r2,
        }

        print(f"{name:16s} {rmse:8.3f} {r2:8.3f}")

    return models, benchmark, X_test, y_test


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--features",
        default="outputs/segment_features.csv",
    )

    ap.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    ap.add_argument(
        "--target-col",
        default=None,
        help="column containing a real historical risk/priority target",
    )

    ap.add_argument(
        "--force-model",
        default=None,
        choices=["xgboost", "lightgbm", "random_forest"],
        help="force a specific available model instead of automatic selection",
    )

    ap.add_argument(
        "--out",
        default="outputs/segment_risk_scores.csv",
    )

    args = ap.parse_args()

    # ---------------------------------------------------------
    # Load configuration and features
    # ---------------------------------------------------------
    cfg = load_config(args.config)
    df = pd.read_csv(args.features)

    # ---------------------------------------------------------
    # Target
    # ---------------------------------------------------------
    if args.target_col and args.target_col in df.columns:
        target_col = args.target_col

        print(
            f"Using real supplied target column: {target_col}"
        )

    else:
        target_col = "risk_score"

        df[target_col] = build_proxy_target(df)

        print(
            "[info] No real historical target supplied — using a documented "
            "PROXY target (defect frequency + severity + coverage). Replace "
            "with real maintenance/IRI records for production-grade validation."
        )

    # ---------------------------------------------------------
    # Prepare features
    # ---------------------------------------------------------
    X, y, feature_cols = prepare_features(
        df,
        target_col,
    )

    # ---------------------------------------------------------
    # Train and benchmark
    # ---------------------------------------------------------
    models, benchmark, X_test, y_test = train_models(
        X,
        y,
        cfg,
    )

    # ---------------------------------------------------------
    # Select model
    # ---------------------------------------------------------
    if args.force_model:

        if args.force_model not in models:
            available = ", ".join(models.keys())

            raise ValueError(
                f"Requested model '{args.force_model}' is not available. "
                f"Available models: {available}"
            )

        best_name = args.force_model
        best_model = models[best_name]

        print(
            f"\n[info] --force-model set: using {best_name}"
        )

    else:

        best_name = min(
            benchmark,
            key=lambda name: benchmark[name]["rmse"],
        )

        best_model = models[best_name]

        print(
            f"\nBest model by RMSE: {best_name}"
        )

    # ---------------------------------------------------------
    # Predict all segments
    # ---------------------------------------------------------
    df["predicted_risk_score"] = (
        best_model.predict(X).round(2)
    )

    # If using the proxy target, preserve the proxy as risk_score.
    # If using a real target, predicted values become the risk score.
    if target_col != "risk_score":
        df["risk_score"] = df["predicted_risk_score"]

    df["model_used"] = best_name

    # ---------------------------------------------------------
    # Priority tiers
    # ---------------------------------------------------------
    thresholds = cfg.get(
        "hotspot_thresholds",
        {
            "high": 66,
            "moderate": 33,
        },
    )

    def tier(score):
        if score >= thresholds["high"]:
            return "high"

        if score >= thresholds["moderate"]:
            return "moderate"

        return "normal"

    df["priority_tier"] = df["risk_score"].apply(tier)

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    out_cols = [
        "road_name",
        "segment_id",
        "chainage_start_m",
        "latitude",
        "longitude",
        "defect_count",
        "dominant_class",
        "mean_severity",
        "max_severity",
        "risk_score",
        "priority_tier",
        "model_used",
    ]

    out_cols = [
        c for c in out_cols
        if c in df.columns
    ]

    Path(args.out).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df[out_cols].to_csv(
        args.out,
        index=False,
    )

    print(
        f"\nWrote {len(df)} segment risk scores -> {args.out}"
    )

    print(
        df["priority_tier"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()
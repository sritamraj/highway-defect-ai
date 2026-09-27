"""
Aggregate point-level geotagged detections into per-road-segment features,
the input table for the predictive maintenance risk model.

A "segment" is a fixed-length chainage bucket (default 100 m, set in
config.yaml under gis.segment_length_m).

Features per segment:
    defect_count            total detections in the segment
    unique_classes          number of distinct defect classes present
    mean_severity           average severity weight of detections
    max_severity            worst single defect's severity
    mean_confidence         average detector confidence (data-quality proxy)
    dominant_class          most frequent defect class in the segment
    pothole_count, ... _count   per-class counts (one column per class)
    prior_visits            number of separate passes over this segment (if
                             the log spans multiple dates/videos) — a proxy
                             for how well-observed the segment is
    traffic_aadt, rainfall_mm   optional covariates if a lookup table is
                             supplied (left NaN otherwise; the model treats
                             missing covariates as "unknown", not zero)

This is intentionally simple and transparent — the point is that a
well-chosen small feature set beats over-engineering here.

Usage:
    python src/predictive/feature_engineering.py --geotagged outputs/geotagged_detections.csv \
        --config configs/config.yaml --out outputs/segment_features.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config


def build_segment_features(df: pd.DataFrame, segment_length_m: float,
                            covariates: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.dropna(subset=["latitude", "longitude", "chainage_m"]).copy()
    if df.empty:
        raise ValueError("No geotagged rows with valid chainage — cannot build segments.")

    df["segment_id"] = (df["chainage_m"] // segment_length_m).astype(int)

    class_dummies = pd.get_dummies(df["class_name"], prefix="", prefix_sep="").add_suffix("_count")

    agg = df.groupby(["road_name", "segment_id"]).agg(
        defect_count=("class_name", "count"),
        unique_classes=("class_name", "nunique"),
        mean_severity=("severity", "mean"),
        max_severity=("severity", "max"),
        mean_confidence=("confidence", "mean"),
        chainage_start_m=("chainage_m", "min"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
    ).reset_index()

    class_counts = pd.concat([df[["road_name", "segment_id"]], class_dummies], axis=1) \
                     .groupby(["road_name", "segment_id"]).sum().reset_index()

    dominant = df.groupby(["road_name", "segment_id"])["class_name"] \
                 .agg(lambda s: s.value_counts().idxmax()).reset_index() \
                 .rename(columns={"class_name": "dominant_class"})

    prior_visits = pd.Series(1, index=agg.index)  # single-pass demo default
    if "visit_date" in df.columns:
        prior_visits = df.groupby(["road_name", "segment_id"])["visit_date"].nunique().values

    features = agg.merge(class_counts, on=["road_name", "segment_id"]) \
                   .merge(dominant, on=["road_name", "segment_id"])
    features["prior_visits"] = prior_visits

    if covariates is not None:
        features = features.merge(covariates, on=["road_name", "segment_id"], how="left")
        for col in ["traffic_aadt", "rainfall_mm"]:
            if col not in features.columns:
                features[col] = np.nan

    return features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geotagged", default="outputs/geotagged_detections.csv")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--covariates", default=None,
                     help="optional CSV with road_name, segment_id, traffic_aadt, rainfall_mm")
    ap.add_argument("--out", default="outputs/segment_features.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.geotagged)
    covariates = pd.read_csv(args.covariates) if args.covariates else None

    seg_len = cfg.get("gis", {}).get("segment_length_m", 100)
    features = build_segment_features(df, seg_len, covariates)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, index=False)
    print(f"Built {len(features)} segment feature rows -> {args.out}")


if __name__ == "__main__":
    main()

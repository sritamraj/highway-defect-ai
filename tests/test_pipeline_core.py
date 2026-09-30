import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gis.geotag import build_chainage, geotag, haversine_m
from src.predictive.feature_engineering import build_segment_features
from src.predictive.risk_model import build_proxy_target, prepare_features


def test_haversine_distance():
    distance = haversine_m(20.0, 85.0, 20.001, 85.0)

    assert distance > 0
    assert distance < 200


def test_build_chainage_starts_at_zero():
    gps = pd.DataFrame({
        "timestamp_s": [0.0, 1.0, 2.0],
        "latitude": [20.0, 20.001, 20.002],
        "longitude": [85.0, 85.0, 85.0],
    })

    result = build_chainage(gps)

    assert "chainage_m" in result.columns
    assert result["chainage_m"].iloc[0] == 0
    assert result["chainage_m"].is_monotonic_increasing


def test_geotag_produces_coordinates():
    detections = pd.DataFrame({
        "frame_id": [0, 1],
        "timestamp_s": [0.5, 1.5],
        "class_name": ["D11", "D20"],
        "confidence": [0.8, 0.7],
        "x1": [10.0, 20.0],
        "y1": [10.0, 20.0],
        "x2": [30.0, 40.0],
        "y2": [30.0, 40.0],
    })

    gps = pd.DataFrame({
        "timestamp_s": [0.0, 1.0, 2.0],
        "latitude": [20.0, 20.001, 20.002],
        "longitude": [85.0, 85.001, 85.002],
    })

    cfg = {
        "gis": {
            "max_gps_interp_gap_s": 5,
        },
        "severity_weights": {
            "D11": 0.5,
            "D20": 0.5,
        },
    }

    result = geotag(
        detections,
        gps,
        cfg,
        road_name="TEST-ROAD",
        max_gap_s=5,
    )

    assert len(result) == 2
    assert result["latitude"].notna().all()
    assert result["longitude"].notna().all()
    assert result["chainage_m"].notna().all()
    assert result["road_name"].eq("TEST-ROAD").all()
    assert result["severity"].notna().all()


def test_segment_features_have_required_columns():
    detections = pd.DataFrame({
        "road_name": ["TEST-ROAD", "TEST-ROAD", "TEST-ROAD"],
        "timestamp_s": [1.0, 2.0, 3.0],
        "class_name": ["D11", "D20", "D11"],
        "confidence": [0.8, 0.7, 0.9],
        "chainage_m": [10.0, 20.0, 110.0],
        "latitude": [20.0, 20.0, 20.001],
        "longitude": [85.0, 85.001, 85.002],
        "severity": [0.5, 0.5, 0.5],
    })

    result = build_segment_features(
        detections,
        segment_length_m=100,
    )

    required = {
        "road_name",
        "segment_id",
        "defect_count",
        "unique_classes",
        "mean_severity",
        "max_severity",
        "mean_confidence",
        "chainage_start_m",
        "latitude",
        "longitude",
    }

    assert required.issubset(result.columns)
    assert len(result) == 2
    assert result["defect_count"].sum() == 3


def test_proxy_target_is_finite():
    features = pd.DataFrame({
        "defect_count": [1, 5, 10],
        "unique_classes": [1, 2, 3],
        "mean_severity": [0.3, 0.5, 0.8],
        "max_severity": [0.4, 0.6, 1.0],
        "mean_confidence": [0.5, 0.7, 0.9],
        "prior_visits": [1, 2, 3],
    })

    target = build_proxy_target(features)

    assert len(target) == 3
    assert target.notna().all()
    assert target.apply(pd.api.types.is_number).all()


def test_prepare_features_returns_numeric_data():
    features = pd.DataFrame({
        "defect_count": [1, 5],
        "unique_classes": [1, 2],
        "mean_severity": [0.3, 0.5],
        "max_severity": [0.4, 0.6],
        "mean_confidence": [0.5, 0.7],
        "dominant_class": ["D11", "D20"],
        "risk_score": [10.0, 20.0],
    })

    X, y, feature_names = prepare_features(
        features,
        target_col="risk_score",
    )

    assert len(X) == len(y)
    assert len(X) == 2
    assert len(feature_names) == X.shape[1]
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)
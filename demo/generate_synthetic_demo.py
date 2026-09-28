"""
Generates a synthetic 'drive' along a fake highway stretch: a GPS log plus
plausible defect detections, so the full pipeline (geotag -> features ->
risk model -> hotspot map) can be run and inspected before any real imagery
or a trained model is available.

Run this first if you just cloned the repo and want to see the pipeline work:
    python demo/generate_synthetic_demo.py
    python src/pipeline/run_pipeline.py --config configs/config.yaml
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

CLASSES = [
    "pothole", "longitudinal_crack", "transverse_crack", "alligator_crack",
    "damaged_lane_marking", "damaged_road_sign", "damaged_crash_barrier", "water_stagnation",
]
# rough real-world-like imbalance: cracks and lane markings are common, barrier damage is rare
CLASS_WEIGHTS = np.array([0.18, 0.24, 0.16, 0.10, 0.16, 0.08, 0.03, 0.05])
CLASS_WEIGHTS = CLASS_WEIGHTS / CLASS_WEIGHTS.sum()

# a fake ~6 km stretch near Bhubaneswar, Odisha, driven at ~40 km/h for ~9 minutes
START_LAT, START_LON = 20.2961, 85.8245
N_GPS_POINTS = 600          # ~1 fix/sec for 10 min
N_DETECTIONS = 350
TOTAL_DURATION_S = 600


def make_gps_log() -> pd.DataFrame:
    t = np.linspace(0, TOTAL_DURATION_S, N_GPS_POINTS)
    # gentle curving road: mostly northeast with a wobble
    lat = START_LAT + 0.00003 * t + 0.0004 * np.sin(t / 80)
    lon = START_LON + 0.00004 * t + 0.0003 * np.cos(t / 120)
    return pd.DataFrame({"timestamp_s": t, "latitude": lat, "longitude": lon})


def make_detections(gps_log: pd.DataFrame) -> pd.DataFrame:
    ts = np.sort(RNG.uniform(0, TOTAL_DURATION_S, N_DETECTIONS))
    classes = RNG.choice(CLASSES, size=N_DETECTIONS, p=CLASS_WEIGHTS)
    conf = np.clip(RNG.normal(0.72, 0.12, N_DETECTIONS), 0.35, 0.98)

    # cluster some defects into 3 "bad" hotspot windows to make the demo map interesting
    hotspot_centers = RNG.choice(ts, size=3, replace=False)
    for c in hotspot_centers:
        extra_n = RNG.integers(15, 30)
        extra_ts = np.clip(RNG.normal(c, 8, extra_n), 0, TOTAL_DURATION_S)
        extra_classes = RNG.choice(
            ["pothole", "alligator_crack", "water_stagnation"], size=extra_n, p=[0.5, 0.3, 0.2]
        )
        extra_conf = np.clip(RNG.normal(0.8, 0.08, extra_n), 0.4, 0.99)
        ts = np.concatenate([ts, extra_ts])
        classes = np.concatenate([classes, extra_classes])
        conf = np.concatenate([conf, extra_conf])

    order = np.argsort(ts)
    ts, classes, conf = ts[order], classes[order], conf[order]

    n = len(ts)
    x1 = RNG.uniform(0, 550, n); y1 = RNG.uniform(0, 350, n)
    x2 = x1 + RNG.uniform(30, 120, n); y2 = y1 + RNG.uniform(20, 90, n)

    return pd.DataFrame({
        "frame_id": np.arange(n),
        "timestamp_s": ts.round(2),
        "class_name": classes,
        "confidence": conf.round(3),
        "x1": x1.round(1), "y1": y1.round(1), "x2": x2.round(1), "y2": y2.round(1),
    })


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    gps_log = make_gps_log()
    detections = make_detections(gps_log)

    gps_log.to_csv(data_dir / "gps_log.csv", index=False)
    detections.to_csv(out_dir / "detections.csv", index=False)

    print(f"Synthetic GPS log:   {data_dir/'gps_log.csv'}  ({len(gps_log)} fixes)")
    print(f"Synthetic detections: {out_dir/'detections.csv'}  ({len(detections)} detections)")
    print("\nThis is FAKE data for demonstrating the pipeline end-to-end. "
          "Replace with real detector output + a real GPS log for an actual road survey.")


if __name__ == "__main__":
    main()

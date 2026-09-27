"""
Orchestrates the full pipeline: geo-tagging -> segment features -> risk
scoring -> hotspot map. Assumes detections.csv already exists (either from
src/detection/infer.py on real footage, or from demo/generate_synthetic_demo.py).

Usage:
    python src/pipeline/run_pipeline.py --config configs/config.yaml --demo
    python src/pipeline/run_pipeline.py --config configs/config.yaml --road-name "NH-16"
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--demo", action="store_true", help="generate synthetic demo data first")
    ap.add_argument("--road-name", default="NH-Sample")
    args = ap.parse_args()
    py = sys.executable

    if args.demo:
        run([py, "demo/generate_synthetic_demo.py"])

    run([py, "src/gis/geotag.py",
         "--detections", "outputs/detections.csv",
         "--gps", "data/gps_log.csv",
         "--config", args.config,
         "--road-name", args.road_name,
         "--out", "outputs/geotagged_detections.csv"])

    run([py, "src/predictive/feature_engineering.py",
         "--geotagged", "outputs/geotagged_detections.csv",
         "--config", args.config,
         "--out", "outputs/segment_features.csv"])

    run([py, "src/predictive/risk_model.py",
         "--features", "outputs/segment_features.csv",
         "--config", args.config,
         "--out", "outputs/segment_risk_scores.csv"])

    run([py, "src/gis/map_viz.py", "--config", args.config])

    print("\nPipeline complete. Open outputs/hotspot_map.html in a browser to view the result.")


if __name__ == "__main__":
    main()

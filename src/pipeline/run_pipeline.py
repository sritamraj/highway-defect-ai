"""
Orchestrates the full highway-defect pipeline:

YOLO inference -> geo-tagging -> segment features -> risk scoring -> hotspot map

Examples:
    Demo:
        python src/pipeline/run_pipeline.py --config configs/config.yaml --demo

    Real/model inference:
        python src/pipeline/run_pipeline.py ^
            --config configs/config.yaml ^
            --weights runs/detect/.../weights/best.pt ^
            --source path/to/images ^
            --fps 1.29 ^
            --road-name "NH-Sample"
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
    ap.add_argument(
        "--demo",
        action="store_true",
        help="generate synthetic demo data first",
    )
    ap.add_argument("--road-name", default="NH-Sample")

    ap.add_argument(
        "--weights",
        help="YOLO model weights; required with --source",
    )
    ap.add_argument(
        "--source",
        help="image/video source; required with --weights",
    )
    ap.add_argument(
        "--fps",
        type=float,
        help="assumed FPS for still-image folders",
    )

    args = ap.parse_args()
    py = sys.executable

    if args.weights and not args.source:
        ap.error("--weights requires --source")

    if args.source and not args.weights:
        ap.error("--source requires --weights")

    # ---------------------------------------------------------
    # 1. Generate detections
    # ---------------------------------------------------------
    if args.demo:
        run([
            py,
            "demo/generate_synthetic_demo.py",
        ])

    elif args.weights and args.source:
        infer_cmd = [
            py,
            "src/detection/infer.py",
            "--weights",
            args.weights,
            "--source",
            args.source,
            "--out",
            "outputs/detections.csv",
        ]

        if args.fps is not None:
            infer_cmd.extend([
                "--fps",
                str(args.fps),
            ])

        run(infer_cmd)

    # ---------------------------------------------------------
    # 2. Geo-tag detections
    # ---------------------------------------------------------
    run([
        py,
        "src/gis/geotag.py",
        "--detections",
        "outputs/detections.csv",
        "--gps",
        "data/gps_log.csv",
        "--config",
        args.config,
        "--road-name",
        args.road_name,
        "--out",
        "outputs/geotagged_detections.csv",
    ])

    # ---------------------------------------------------------
    # 3. Build road-segment features
    # ---------------------------------------------------------
    run([
        py,
        "src/predictive/feature_engineering.py",
        "--geotagged",
        "outputs/geotagged_detections.csv",
        "--config",
        args.config,
        "--out",
        "outputs/segment_features.csv",
    ])

    # ---------------------------------------------------------
    # 4. Calculate maintenance-priority scores
    # ---------------------------------------------------------
    run([
        py,
        "src/predictive/risk_model.py",
        "--features",
        "outputs/segment_features.csv",
        "--config",
        args.config,
        "--out",
        "outputs/segment_risk_scores.csv",
    ])

    # ---------------------------------------------------------
    # 5. Generate GIS hotspot map
    # ---------------------------------------------------------
    run([
        py,
        "src/gis/map_viz.py",
        "--config",
        args.config,
    ])

    print(
        "\nPipeline complete. "
        "Open outputs/hotspot_map.html in a browser to view the result."
    )


if __name__ == "__main__":
    main()
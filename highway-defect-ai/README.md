# AI-Driven Highway Defect Detection & Predictive Maintenance

Prototype AI-based highway condition monitoring system, inspired by NHAI's AI-enabled road
monitoring requirements. It detects road defects from imagery using computer vision,
geo-references each detection, visualizes road-condition hotspots on an interactive map,
and produces a data-driven **Maintenance Risk Score** per road segment.

> **Scope note:** this is a prototype built for an M.Tech / internship portfolio, not a
> production system. It targets 8 defect classes on sample/open imagery, not 30+ classes
> across NHAI's full ~40,000 km network. The architecture is designed to scale toward that,
> but no claim is made to NHAI production data — everything here runs on open datasets or
> your own collected imagery.

---

## Pipeline

```
Road video/images
      │
      ▼
[1] Computer Vision (YOLOv8 / RT-DETR)  ──► defect class, bbox, confidence
      │
      ▼
[2] Geo-referencing (GPS log / EXIF / interpolation) ──► lat, lon, chainage
      │
      ▼
[3] GIS Hotspot Mapping (GeoPandas + Folium) ──► red/yellow/green cluster map
      │
      ▼
[4] Predictive Maintenance (XGBoost / LightGBM) ──► per-segment Maintenance Risk Score
```

## Defect classes (Phase 1 — start small, scale later)

| # | Class                  |
|---|------------------------|
| 1 | pothole                |
| 2 | longitudinal_crack     |
| 3 | transverse_crack       |
| 4 | alligator_crack        |
| 5 | damaged_lane_marking   |
| 6 | damaged_road_sign      |
| 7 | damaged_crash_barrier  |
| 8 | water_stagnation       |

## Repository layout

```
highway-defect-ai/
├── configs/
│   ├── config.yaml            # global paths, thresholds, weights
│   └── yolo_data.yaml         # YOLO dataset spec (classes, splits)
├── src/
│   ├── detection/
│   │   ├── train.py           # transfer-learning training loop (Ultralytics YOLO)
│   │   ├── infer.py           # run detector on images/video → detections.csv
│   │   └── evaluate.py        # precision/recall/F1/mAP50/mAP50-95/FPS/per-class/confusion
│   ├── gis/
│   │   ├── geotag.py          # attach lat/lon/chainage/timestamp to each detection
│   │   └── map_viz.py         # Folium hotspot map (red/yellow/green clustering)
│   ├── predictive/
│   │   ├── feature_engineering.py  # per-segment features from detections
│   │   └── risk_model.py           # XGBoost/LightGBM/RandomForest risk scorer
│   ├── pipeline/
│   │   └── run_pipeline.py    # orchestrates steps 1→4 end to end
│   └── utils/
│       └── config.py          # config loader
├── demo/
│   └── generate_synthetic_demo.py  # makes fake detections/GPS/segments so the
│                                    # whole pipeline runs with zero real data
├── data/                        # put your images/video + GPS log here (gitignored)
├── outputs/                     # detections.csv, risk_scores.csv, hotspot_map.html
├── requirements.txt
└── LICENSE
```

## Quickstart (no real dataset needed yet)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Generate synthetic demo data (detections + GPS log + segments)
python demo/generate_synthetic_demo.py

# 2. Run the full pipeline: geotag → hotspot map → risk scores
python src/pipeline/run_pipeline.py --config configs/config.yaml --demo

# Outputs:
#   outputs/geotagged_detections.csv
#   outputs/hotspot_map.html         <- open in a browser
#   outputs/segment_risk_scores.csv
```

## Quickstart (with real imagery)

1. Collect/label images with [LabelImg](https://github.com/HumanSignal/labelImg) or
   [Roboflow](https://roboflow.com) in YOLO format, or use an open dataset such as
   **RDD2022** (Road Damage Dataset) or **CrackForest**.
2. Point `configs/yolo_data.yaml` at your `train/val/test` splits.
3. Train:
   ```bash
   python src/detection/train.py --data configs/yolo_data.yaml --epochs 100 --model yolov8s.pt
   ```
4. Evaluate:
   ```bash
   python src/detection/evaluate.py --weights runs/detect/train/weights/best.pt --data configs/yolo_data.yaml
   ```
5. Run inference on your drive footage (with a synced GPS log, e.g. from a dashcam + GPX):
   ```bash
   python src/detection/infer.py --weights best.pt --source data/drive_video.mp4 --out outputs/detections.csv
   ```
6. Geo-reference, map, and score exactly as in the demo (`run_pipeline.py` without `--demo`).

## Evaluation reported (not just "accuracy")

`src/detection/evaluate.py` reports, per the standard object-detection protocol:

- Precision, Recall, F1 (per class + macro-averaged)
- mAP@50 and mAP@50:95
- Inference FPS (on the eval hardware)
- Confusion matrix + top confused-class pairs
- Per-class AP bar chart

## Predictive Maintenance Risk Score

For each road segment (chainage bucket, e.g. every 100 m), `risk_model.py` builds:

`defect_frequency + severity-weighted defect count + historical trend (if multiple passes exist) + traffic/weather covariates (optional, if available)` → **Maintenance Risk Score (0–100)**

Three model families are benchmarked (`--model` flag): `xgboost`, `lightgbm`, `random_forest`.
The training script reports RMSE/R² per model on a held-out split and picks the best —
this is deliberately not deep learning, since tabular segment-level data of this size
favors gradient-boosted trees, and the report says so explicitly (this is the "research
judgment" point worth making in your writeup).

## Tech stack

- **Detection:** Ultralytics YOLOv8 (swap-compatible with RT-DETR)
- **GIS:** GeoPandas, Shapely, Folium (Leaflet under the hood)
- **Predictive modeling:** XGBoost, LightGBM, scikit-learn
- **Data:** pandas, numpy

## Suggested thesis/report narrative

IIT Patna → M.Tech AI & DS → Deep Learning → Computer Vision → Geospatial Analytics →
Infrastructure AI → Highway Safety & Maintenance → NHAI-aligned prototype.

## License

MIT — see `LICENSE`.

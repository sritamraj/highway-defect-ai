# AI-Driven Highway Defect Detection and Predictive Maintenance

## Overview

This project presents a prototype AI-based highway condition monitoring system that combines:

* Computer vision for highway defect detection
* YOLO-based object detection
* Quantitative model evaluation
* Geospatial visualization
* Road-segment condition aggregation
* Maintenance-priority scoring

The project is designed as an academic/research prototype for demonstrating how computer vision and geospatial analytics can be combined for highway inspection and predictive-maintenance workflows.

> **Important:** The geospatial coordinates currently used in the prototype are synthetic/demo coordinates. They do not represent actual NHAI asset locations, internal NHAI data, or real-world maintenance records.

---

## Project Pipeline

```text
Road Images
     |
     v
YOLO-based Defect Detection
     |
     v
Detection Results
     |
     v
Confidence + Defect Severity
     |
     v
Geospatial Association
     |
     v
Road Segment Aggregation
     |
     v
Maintenance Priority Score
     |
     v
Interactive GIS Hotspot Map
```

---

## Computer Vision

The project uses Ultralytics YOLO for road-defect object detection.

The repository contains configuration files for:

* A custom 8-class highway-defect taxonomy
* RDD2022 India 9-class experiments
* RDD2022 India 10-class experiments

### Custom defect classes

| ID | Class                 |
| -: | --------------------- |
|  0 | pothole               |
|  1 | longitudinal_crack    |
|  2 | transverse_crack      |
|  3 | alligator_crack       |
|  4 | damaged_lane_marking  |
|  5 | damaged_road_sign     |
|  6 | damaged_crash_barrier |
|  7 | water_stagnation      |

### RDD2022 classes

The RDD2022 experiments use the dataset's defect identifiers such as:

`D00`, `D01`, `D10`, `D11`, `D20`, `D40`, `D43`, `D44`, and `D50`.

The dataset itself is not included in this repository.

---

## Model Evaluation

Several lightweight CPU-friendly YOLO experiments were performed using the RDD2022 validation data.

| Experiment          | Precision | Recall |  mAP50 | mAP50-95 |
| ------------------- | --------: | -----: | -----: | -------: |
| test-10percent      |    0.0012 | 0.2400 | 0.0021 |   0.0006 |
| test-20percent-5ep  |    0.6068 | 0.0222 | 0.0154 |   0.0045 |
| test-20percent-10ep |    0.4858 | 0.0493 | 0.0150 |   0.0043 |

These results are included as experimental baseline measurements. They should not be interpreted as production-ready model performance.

The current experiments were constrained by CPU compute and limited training duration.

---

## Detection Analysis

A baseline YOLO model was evaluated on a subset of RDD2022 validation images.

Current analysis contains:

* **2,683 detection records**
* **464 images with prediction files**
* Confidence-score distributions
* Per-class detection counts
* Confidence-threshold analysis
* IoU-based detection evaluation
* Model comparison results

The generated detection table is available at:

```text
outputs/detections_yolo_baseline.csv
```

---

## Geospatial Analytics

The detection outputs are transformed into a geospatial representation for highway-segment analysis.

The prototype includes:

* Detection-to-coordinate association
* Road-segment assignment
* Defect severity mapping
* Confidence-weighted priority scoring
* Segment-level aggregation
* Maintenance-priority categorization

The current prototype contains:

* **2,683 georeferenced demo detections**
* **54 demo road segments**

### Important data limitation

The coordinates in:

```text
outputs/gis_detections_georeferenced_demo.csv
```

are synthetically generated for demonstration purposes.

They are explicitly marked:

```text
SYNTHETIC DEMO - NOT REAL GPS
```

Therefore, the GIS results must not be presented as actual highway asset locations or official NHAI inspection data.

---

## Maintenance Priority

The prototype aggregates detections by road segment and calculates a maintenance score using detection confidence and defect severity.

Example output:

```text
outputs/maintenance_priority_segments.csv
```

The resulting priority categories are:

* Low
* Medium
* High

The maintenance score is a research prototype indicator created for demonstrating the end-to-end pipeline. It is **not an official NHAI maintenance methodology**.

For a production system, the score should be calibrated using real inspection history, maintenance records, road-condition measurements, traffic exposure, weather, and engineering standards.

---

## GIS Hotspot Map

An interactive Folium-based map is generated at:

```text
outputs/highway_defect_hotspot_map.html
```

The map provides:

* Road-segment locations
* Maintenance-priority markers
* Defect aggregation
* Heatmap visualization
* Interactive geographic exploration

Open the file in a web browser to explore the prototype GIS visualization.

---

## Generated Charts

The repository contains visual summaries in:

```text
outputs/charts/
```

Available charts include:

* `defect_distribution.png`
* `maintenance_priority.png`
* `model_comparison.png`

These provide quick visual summaries of defect detections, segment-priority distribution, and model evaluation experiments.

---

## Project Structure

```text
highway-defect-ai/
|
+-- configs/
|   +-- config.yaml
|   +-- yolo_data.yaml
|   +-- yolo_rdd2022_9class.yaml
|   +-- yolo_rdd2022_10class.yaml
|
+-- demo/
|   +-- generate_synthetic_demo.py
|
+-- outputs/
|   +-- charts/
|   +-- detections_yolo_baseline.csv
|   +-- gis_detections_demo.csv
|   +-- gis_detections_georeferenced_demo.csv
|   +-- maintenance_priority_segments.csv
|   +-- model_comparison.csv
|   +-- highway_defect_hotspot_map.html
|
+-- src/
|   +-- detection/
|   |   +-- evaluate.py
|   |   +-- infer.py
|   |   +-- train.py
|   |
|   +-- gis/
|   |   +-- geotag.py
|   |   +-- map_viz.py
|   |
|   +-- pipeline/
|   |   +-- run_pipeline.py
|   |
|   +-- predictive/
|       +-- feature_engineering.py
|       +-- risk_model.py
|   |
|   +-- utils/
|       +-- config.py
|
+-- requirements.txt
+-- README.md
+-- LICENSE
+-- .gitignore
```

---

## Technologies

* Python
* Ultralytics YOLO
* PyTorch
* OpenCV
* Pandas
* NumPy
* Matplotlib
* Folium
* YAML configuration
* Git/GitHub

---

## Installation

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

The project has been tested with the dependency versions specified in `requirements.txt`.

---

## Dataset

The computer-vision experiments use the Road Damage Detection Dataset (RDD2022), including the India subset.

The dataset is intentionally **not included in this GitHub repository** because of repository size and dataset-distribution considerations.

Place the dataset locally and update the relevant YAML configuration if necessary.

For the RDD2022 9-class experiment, the configuration file is:

```text
configs/yolo_rdd2022_9class.yaml
```

---

## Reproducibility

The repository stores:

* Dataset configuration files
* Training scripts
* Detection scripts
* Evaluation scripts
* Geospatial-processing code
* Predictive-maintenance code
* Generated analytical outputs
* Model-comparison results

Large datasets, trained model weights, and Ultralytics training runs are excluded through `.gitignore`.

---

## Research Scope

This project demonstrates an end-to-end research workflow:

1. Detect road defects from images.
2. Quantify detection confidence and defect type.
3. Associate detections with geographic coordinates.
4. Aggregate defects by road segment.
5. Calculate a prototype maintenance-priority indicator.
6. Visualize potential hotspots using GIS.

The architecture can be extended toward a larger intelligent transportation-system platform.

---

## Future Work

Potential extensions include:

* Training on a larger and more balanced dataset
* Improving rare-defect detection
* Hyperparameter optimization
* Real GPS/INS integration
* Road-network matching
* Temporal deterioration modeling
* Weather and traffic integration
* Real maintenance-history labels
* Pavement-condition indicators
* Edge deployment using optimized models
* Automated inspection-report generation
* Real-time dashboard integration

---

## Disclaimer

This project is an academic/research prototype.

The current geospatial dataset contains synthetic demonstration coordinates and does not represent actual NHAI road locations or internal NHAI operational data.

The maintenance-priority score is a prototype analytical indicator and is not an official NHAI methodology.

Model performance is experimental and should not be interpreted as production-level highway inspection accuracy.

---

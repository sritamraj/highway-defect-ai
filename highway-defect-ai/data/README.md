# Data directory

This folder is gitignored except for this file — real imagery, video, GPS logs,
and trained weights should not be committed to the repo.

## What goes here

- `yolo_dataset/` — your YOLO-format dataset (`images/{train,val,test}`,
  `labels/{train,val,test}`), matching `configs/yolo_data.yaml`.
- `drive_video.mp4` (or a folder of frames) — footage to run inference on.
- `gps_log.csv` — columns `timestamp_s, latitude, longitude`, synced to the
  video's clock. If you only have a GPX track, convert it with `gpxpy`
  (included in requirements.txt).

## Suggested open datasets to bootstrap Phase 1

- **RDD2022 / RDD2020 (Road Damage Dataset)** — multi-country road damage
  images with crack/pothole annotations; good starting point for
  crack/pothole classes.
- **CrackForest** — pavement crack segmentation dataset.
- **Roboflow Universe** — search "pothole detection" or "road damage" for
  several community-labeled YOLO-format datasets you can merge/relabel into
  the 8-class scheme in `configs/yolo_data.yaml`.

Always check each dataset's license before using it in a report or public repo.

## No real data yet?

Run `python demo/generate_synthetic_demo.py` from the repo root — it writes a
fake `gps_log.csv` here and a fake `detections.csv` to `outputs/`, so the rest
of the pipeline (geo-tagging, hotspot map, risk scoring) is fully runnable and
inspectable before you have a trained model or real footage.

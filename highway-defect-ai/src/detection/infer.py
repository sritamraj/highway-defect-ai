"""
Run a trained detector over images or video and write one row per detection
to a CSV that downstream GIS geo-tagging consumes.

Output columns:
    frame_id, timestamp_s, class_name, confidence, x1, y1, x2, y2

`timestamp_s` is seconds from the start of the video (or None for a folder of
still images, unless --fps is given to fake a constant frame rate) — the GIS
step matches this to the nearest GPS log timestamp.

Usage:
    python src/detection/infer.py --weights runs/detect/train/weights/best.pt \
        --source data/drive_video.mp4 --out outputs/detections.csv --conf 0.35
"""
import argparse
import csv
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="video file, image, or folder of images")
    ap.add_argument("--out", default="outputs/detections.csv")
    ap.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--fps", type=float, default=None,
                     help="assumed FPS for a still-image folder, to synthesize timestamps")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = model.names

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []

    t0 = time.time()
    frame_idx = 0
    results = model.predict(source=args.source, conf=args.conf, iou=args.iou, stream=True)

    for r in results:
        # Video frames carry a native timestamp from Ultralytics' frame counter;
        # for images we fall back to an optional synthetic FPS.
        ts = frame_idx / args.fps if args.fps else None
        boxes = r.boxes
        if boxes is not None:
            for b in boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                rows.append({
                    "frame_id": frame_idx,
                    "timestamp_s": ts,
                    "class_name": names[cls_id],
                    "confidence": round(conf, 4),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
        frame_idx += 1

    elapsed = time.time() - t0
    fps = frame_idx / elapsed if elapsed > 0 else 0.0

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_id", "timestamp_s", "class_name", "confidence", "x1", "y1", "x2", "y2"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} detections across {frame_idx} frames to {args.out}")
    print(f"Inference throughput: {fps:.1f} FPS (includes I/O; see evaluate.py for a clean benchmark)")


if __name__ == "__main__":
    main()

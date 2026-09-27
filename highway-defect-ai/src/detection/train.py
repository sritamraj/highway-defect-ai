"""
Train a YOLOv8 (or RT-DETR) detector on the highway-defect dataset using
transfer learning from COCO-pretrained weights.

Usage:
    python src/detection/train.py --data configs/yolo_data.yaml \
        --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16

Notes on class balancing:
    Ultralytics applies mosaic/mixup augmentation and its loss already
    down-weights easy negatives, but with only 8 classes and a likely
    long-tailed class distribution (potholes >> damaged crash barriers),
    also:
      1. Oversample rare-class images at the dataset level (see
         `compute_class_weights` below to inspect the imbalance first).
      2. Keep mosaic augmentation ON (default) to synthetically increase
         rare-class exposure per epoch.
      3. Consider a focal-loss variant if a class stays under-recalled
         after step 1-2 (Ultralytics supports this via `--cls-loss focal`
         in recent versions; check your installed version's docs).
"""
import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml


def compute_class_weights(data_yaml: str) -> dict:
    """Counts label occurrences across the train split to reveal class imbalance."""
    cfg = yaml.safe_load(open(data_yaml))
    root = Path(data_yaml).parent / cfg["path"]
    train_images = root / cfg["train"]
    label_dir = Path(str(train_images).replace("images", "labels"))

    counts = Counter()
    if not label_dir.exists():
        print(f"[warn] label dir {label_dir} not found yet — skipping class-balance report.")
        return {}

    for label_file in label_dir.glob("*.txt"):
        for line in open(label_file):
            cls_id = line.strip().split()[0]
            counts[int(cls_id)] = counts.get(int(cls_id), 0) + 1

    names = cfg.get("names", {})
    total = sum(counts.values()) or 1
    print("\nClass distribution (train split):")
    for cls_id, n in sorted(counts.items()):
        name = names.get(cls_id, str(cls_id))
        print(f"  {name:24s} {n:6d}  ({100*n/total:5.1f}%)")
    return dict(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/yolo_data.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="COCO-pretrained checkpoint to fine-tune from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20, help="early-stopping patience")
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="train")
    args = ap.parse_args()

    compute_class_weights(args.data)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics first: pip install -r requirements.txt")
        sys.exit(1)

    model = YOLO(args.model)  # transfer learning from COCO weights
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        # augmentation (defaults are sensible; tuned slightly for road scenes)
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # lighting/weather variance
        degrees=5.0, translate=0.1, scale=0.5, shear=2.0,
        fliplr=0.5, flipud=0.0,               # roads aren't flipped vertically
        # calibration-friendly settings
        cos_lr=True,
        label_smoothing=0.05,
    )
    print(f"\nTraining complete. Weights at: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()

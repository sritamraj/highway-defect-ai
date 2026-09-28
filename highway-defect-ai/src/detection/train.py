"""
Train a YOLOv8 detector on the highway-defect dataset using
COCO-pretrained weights.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml


def compute_class_weights(data_yaml: str) -> dict:
    """Count label occurrences in the training split."""
    cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))

    root = Path(cfg["path"])
    train_images = root / cfg["train"]
    label_dir = Path(str(train_images).replace("images", "labels"))

    counts = Counter()

    if not label_dir.exists():
        print(f"[warn] label directory not found: {label_dir}")
        return {}

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1

    names = cfg.get("names", {})
    total = sum(counts.values()) or 1

    print("\nClass distribution (train split):")
    for cls_id, count in sorted(counts.items()):
        name = names.get(cls_id, str(cls_id))
        print(f"  {name:24s} {count:6d}  ({100 * count / total:5.1f}%)")

    return dict(counts)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        default="configs/yolo_rdd2022_9class.yaml"
    )

    ap.add_argument(
        "--model",
        default="yolov8n.pt",
        help="COCO-pretrained checkpoint"
    )

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patience", type=int, default=5)

    # Use a fraction of the training dataset for CPU-friendly experiments.
    ap.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0 < fraction <= 1)"
    )

    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="train")

    args = ap.parse_args()

    if not 0 < args.fraction <= 1.0:
        print("[error] --fraction must be greater than 0 and <= 1.0")
        sys.exit(1)

    print("\nTraining configuration:")
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {args.data}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch:      {args.batch}")
    print(f"  Fraction:   {args.fraction}")
    print(f"  Run name:   {args.name}")

    compute_class_weights(args.data)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics first: pip install -r requirements.txt")
        sys.exit(1)

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        fraction=args.fraction,
        project=args.project,
        name=args.name,

        # Road-scene augmentation
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        flipud=0.0,

        # Learning-rate schedule
        cos_lr=True,
    )

    print(
        f"\nTraining complete. "
        f"Weights at: {args.project}/{args.name}/weights/best.pt"
    )


if __name__ == "__main__":
    main()
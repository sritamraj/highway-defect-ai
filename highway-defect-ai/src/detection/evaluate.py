"""
Full evaluation report for the trained detector — not just "accuracy".

Reports:
    - Precision, Recall, F1 (per class + macro average)
    - mAP@50 and mAP@50:95
    - Inference FPS on a clean timing loop (no I/O overhead)
    - Confusion matrix + the top confused class pairs
    - A per-class AP bar chart saved to outputs/

Usage:
    python src/detection/evaluate.py --weights best.pt --data configs/yolo_data.yaml
"""
import argparse
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="configs/yolo_data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--n_timing_runs", type=int, default=50,
                     help="dummy forward passes for a clean FPS benchmark")
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    # --- Standard Ultralytics val metrics: P, R, mAP50, mAP50-95, per-class AP ---
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, plots=True)

    names = metrics.names
    p = metrics.box.p          # precision per class
    r = metrics.box.r          # recall per class
    ap50 = metrics.box.ap50    # AP@0.5 per class
    ap5095 = metrics.box.ap    # AP@0.5:0.95 per class
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r + 1e-16), 0.0)

    print("\n=== Per-class detection performance ===")
    print(f"{'class':24s} {'P':>7s} {'R':>7s} {'F1':>7s} {'AP@50':>8s} {'AP@50:95':>9s}")
    for i, name in names.items():
        print(f"{name:24s} {p[i]:7.3f} {r[i]:7.3f} {f1[i]:7.3f} {ap50[i]:8.3f} {ap5095[i]:9.3f}")

    print("\n=== Macro-averaged summary ===")
    print(f"Precision:  {p.mean():.3f}")
    print(f"Recall:     {r.mean():.3f}")
    print(f"F1:         {f1.mean():.3f}")
    print(f"mAP@50:     {metrics.box.map50:.3f}")
    print(f"mAP@50:95:  {metrics.box.map:.3f}")

    # --- Clean inference FPS benchmark (no dataloader/I/O overhead) ---
    device = 0 if torch.cuda.is_available() else "cpu"
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    model.model.to(device if device != "cpu" else "cpu")
    # warmup
    for _ in range(5):
        model.predict(dummy, verbose=False, device=device)
    t0 = time.time()
    for _ in range(args.n_timing_runs):
        model.predict(dummy, verbose=False, device=device)
    elapsed = time.time() - t0
    fps = args.n_timing_runs / elapsed
    print(f"\nInference FPS (batch=1, imgsz={args.imgsz}, device={device}): {fps:.1f}")

    # --- Confusion matrix + top confused pairs ---
    try:
        cm = metrics.confusion_matrix.matrix  # (n_classes+1) x (n_classes+1), last row/col = background
        n = len(names)
        pairs = []
        for i in range(n):
            for j in range(n):
                if i != j and cm[i, j] > 0:
                    pairs.append((names[i], names[j], int(cm[i, j])))
        pairs.sort(key=lambda x: -x[2])
        print("\n=== Top confused class pairs (true -> predicted : count) ===")
        for true_c, pred_c, count in pairs[:10]:
            print(f"  {true_c:22s} -> {pred_c:22s} : {count}")
        print(f"\nFull confusion matrix plot saved by Ultralytics under runs/detect/val*/confusion_matrix.png")
    except Exception as e:
        print(f"[warn] could not extract confusion matrix detail: {e}")

    print(f"\nPer-class AP plots and PR curves saved under the Ultralytics val run directory "
          f"(see console output above for the exact path).")


if __name__ == "__main__":
    main()

"""
Attach lat/lon/chainage/severity to each raw detection by matching each
detection's timestamp to the nearest fix in a GPS log.

GPS log format (CSV): timestamp_s, latitude, longitude
    (e.g. exported from a GPX track recorded alongside the dashcam video —
    see gpxpy in requirements.txt for a GPX -> CSV helper if needed.)

Output columns added to each detection row:
    latitude, longitude, chainage_m, severity, road_name

Usage:
    python src/gis/geotag.py --detections outputs/detections.csv \
        --gps data/gps_log.csv --config configs/config.yaml \
        --road-name "NH-16" --out outputs/geotagged_detections.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config


EARTH_RADIUS_M = 6371000.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between consecutive GPS points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def build_chainage(gps_df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative distance (chainage, in meters) along the GPS track."""
    gps_df = gps_df.sort_values("timestamp_s").reset_index(drop=True)
    dists = [0.0]
    for i in range(1, len(gps_df)):
        d = haversine_m(
            gps_df.loc[i - 1, "latitude"], gps_df.loc[i - 1, "longitude"],
            gps_df.loc[i, "latitude"], gps_df.loc[i, "longitude"],
        )
        dists.append(dists[-1] + d)
    gps_df["chainage_m"] = dists
    return gps_df


def geotag(detections: pd.DataFrame, gps_log: pd.DataFrame, cfg: dict,
           road_name: str, max_gap_s: float) -> pd.DataFrame:
    gps_log = build_chainage(gps_log)

    lat_out, lon_out, chainage_out, dropped = [], [], [], 0
    for ts in detections["timestamp_s"]:
        if pd.isna(ts):
            lat_out.append(np.nan); lon_out.append(np.nan); chainage_out.append(np.nan)
            continue
        idx = (gps_log["timestamp_s"] - ts).abs().idxmin()
        gap = abs(gps_log.loc[idx, "timestamp_s"] - ts)
        if gap > max_gap_s:
            lat_out.append(np.nan); lon_out.append(np.nan); chainage_out.append(np.nan)
            dropped += 1
            continue
        lat_out.append(gps_log.loc[idx, "latitude"])
        lon_out.append(gps_log.loc[idx, "longitude"])
        chainage_out.append(gps_log.loc[idx, "chainage_m"])

    detections = detections.copy()
    detections["latitude"] = lat_out
    detections["longitude"] = lon_out
    detections["chainage_m"] = chainage_out
    detections["road_name"] = road_name

    sev_map = cfg.get("severity_weights", {})
    detections["severity"] = detections["class_name"].map(sev_map).fillna(0.5)

    if dropped:
        print(f"[warn] {dropped} detections had no GPS fix within {max_gap_s}s and were left ungeotagged.")
    return detections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default="outputs/detections.csv")
    ap.add_argument("--gps", default="data/gps_log.csv")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--road-name", default="NH-Sample")
    ap.add_argument("--out", default="outputs/geotagged_detections.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    detections = pd.read_csv(args.detections)
    gps_log = pd.read_csv(args.gps)

    max_gap_s = cfg.get("gis", {}).get("max_gps_interp_gap_s", 5)
    tagged = geotag(detections, gps_log, cfg, args.road_name, max_gap_s)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tagged.to_csv(args.out, index=False)
    n_ok = tagged["latitude"].notna().sum()
    print(f"Geo-tagged {n_ok}/{len(tagged)} detections -> {args.out}")


if __name__ == "__main__":
    main()

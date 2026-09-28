"""
Build an interactive Folium map of road-condition hotspots:
    red    = high-priority segment
    yellow = moderate-priority segment
    green  = normal / low-priority segment

Consumes either:
    (a) geotagged_detections.csv directly (colors each point by severity), or
    (b) segment_risk_scores.csv (colors each segment by Maintenance Risk Score)
if both are available, (b) is preferred since it reflects the full model,
not just raw severity.

Usage:
    python src/gis/map_viz.py --config configs/config.yaml
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import folium
from folium.plugins import MarkerCluster

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config


def score_to_color(score: float, thresholds: dict) -> str:
    if score >= thresholds.get("high", 66):
        return "red"
    if score >= thresholds.get("moderate", 33):
        return "orange"
    return "green"


def build_point_map(df: pd.DataFrame, thresholds: dict, out_path: str, lat_col="latitude",
                     lon_col="longitude", score_col="severity", popup_cols=None):
    df = df.dropna(subset=[lat_col, lon_col])
    if df.empty:
        print("[warn] no geotagged rows to map.")
        return

    center = [df[lat_col].mean(), df[lon_col].mean()]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    cluster = MarkerCluster().add_to(m)

    popup_cols = popup_cols or []
    for _, row in df.iterrows():
        score = row[score_col] * 100 if score_col == "severity" else row[score_col]
        color = score_to_color(score, thresholds)
        popup_lines = [f"<b>{c}:</b> {row[c]}" for c in popup_cols if c in row]
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup("<br>".join(popup_lines), max_width=300) if popup_lines else None,
        ).add_to(cluster)

    _add_legend(m)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(out_path)
    print(f"Hotspot map saved to {out_path}")


def _add_legend(m: folium.Map):
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border: 1px solid #999;
                border-radius: 6px; font-size: 13px; box-shadow: 0 1px 4px rgba(0,0,0,.3);">
      <b>Maintenance Priority</b><br>
      <span style="color:red;">&#9679;</span> High &nbsp;
      <span style="color:orange;">&#9679;</span> Moderate &nbsp;
      <span style="color:green;">&#9679;</span> Normal
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--geotagged", default=None, help="override geotagged detections path")
    ap.add_argument("--risk-scores", default=None, help="override risk scores path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    thresholds = cfg.get("hotspot_thresholds", {})
    paths = cfg.get("paths", {})

    risk_path = Path(args.risk_scores or paths.get("risk_scores", "outputs/segment_risk_scores.csv"))
    geotag_path = Path(args.geotagged or paths.get("geotagged_detections", "outputs/geotagged_detections.csv"))
    out_path = paths.get("hotspot_map", "outputs/hotspot_map.html")

    if risk_path.exists():
        df = pd.read_csv(risk_path)
        build_point_map(
            df, thresholds, out_path,
            lat_col="latitude", lon_col="longitude", score_col="risk_score",
            popup_cols=["road_name", "chainage_m", "risk_score", "defect_count", "dominant_class"],
        )
    elif geotag_path.exists():
        df = pd.read_csv(geotag_path)
        build_point_map(
            df, thresholds, out_path,
            lat_col="latitude", lon_col="longitude", score_col="severity",
            popup_cols=["road_name", "class_name", "confidence", "chainage_m"],
        )
    else:
        print("[error] neither risk scores nor geotagged detections found. Run the pipeline first.")


if __name__ == "__main__":
    main()

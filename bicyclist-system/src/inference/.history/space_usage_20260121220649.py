# src/inference/space_usage.py
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd

from src.scene.roi import ROIMaskEngine
from src.inference.common import FrameLabelThresholds, label_track_frames

@dataclass
class SpaceUsageThresholds(FrameLabelThresholds):
    min_track_frames: int = 5
    dominant_min_ratio: float = 0.40

def infer_space_usage_per_track(
    tracks_df: pd.DataFrame,
    roi: ROIMaskEngine,
    thr: SpaceUsageThresholds = SpaceUsageThresholds(),
) -> Tuple[pd.DataFrame, Dict]:
    required = {"track_id", "x1", "y1", "x2", "y2"}
    missing = required - set(tracks_df.columns)
    if missing:
        raise ValueError(f"tracks_df missing columns: {missing}")

    rows = []
    for tid, g in tracks_df.groupby("track_id", sort=False):
        if len(g) < thr.min_track_frames:
            rows.append({
                "track_id": tid,
                "in_bike_lane_any": False,
                "in_sidewalk_any": False,
                "in_roadway_any": False,
                "dominant_space": "unknown",
            })
            continue

        labels = label_track_frames(g, roi, thr)
        n = len(labels)
        counts = {
            "bike_lane": labels.count("bike_lane"),
            "sidewalk": labels.count("sidewalk"),
            "roadway": labels.count("roadway"),
            "crosswalk": labels.count("crosswalk"),
        }

        in_bike_any = counts["bike_lane"] > 0
        in_side_any = counts["sidewalk"] > 0
        in_road_any = (counts["roadway"] > 0) or (counts["crosswalk"] > 0)  # crossing is roadway semantics

        # dominant: by ratio, with priority crosswalk>bikelane>sidewalk>roadway if ties
        ratios = {k: (v / n) for k, v in counts.items()}
        best_label = max(["crosswalk", "bike_lane", "sidewalk", "roadway"], key=lambda k: (ratios[k], ["crosswalk","bike_lane","sidewalk","roadway"].index(k)*-1))
        dominant = best_label if ratios[best_label] >= thr.dominant_min_ratio else "unknown"
        if dominant == "crosswalk":
            dominant = "roadway"  # optional: keep your output set consistent (bike/side/road/unknown)

        rows.append({
            "track_id": tid,
            "in_bike_lane_any": bool(in_bike_any),
            "in_sidewalk_any": bool(in_side_any),
            "in_roadway_any": bool(in_road_any),
            "dominant_space": dominant,
        })

    events_df = pd.DataFrame(rows)
    summary = {
        "N_tracks": int(events_df.shape[0]),
        "N_dom_bike_lane": int((events_df["dominant_space"] == "bike_lane").sum()) if len(events_df) else 0,
        "N_dom_sidewalk": int((events_df["dominant_space"] == "sidewalk").sum()) if len(events_df) else 0,
        "N_dom_roadway": int((events_df["dominant_space"] == "roadway").sum()) if len(events_df) else 0,
        "N_dom_unknown": int((events_df["dominant_space"] == "unknown").sum()) if len(events_df) else 0,
    }
    return events_df, summary

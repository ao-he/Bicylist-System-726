# src/inference/right_side.py
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pandas as pd

from src.scene.roi import ROIMaskEngine
from src.inference.common import FrameLabelThresholds, label_track_frames, direction_cosine

@dataclass
class RightSideThresholds(FrameLabelThresholds):
    min_track_frames: int = 5
    min_move_px: float = 8.0
    band_label: str = "bike_lane"   # minimal change: keep your current semantics
    road_label: str = "roadway"     # roadway semantics

def infer_right_side_per_track(
    tracks_df: pd.DataFrame,
    roi: ROIMaskEngine,
    thr: RightSideThresholds = RightSideThresholds(),
) -> Tuple[pd.DataFrame, Dict]:
    required = {"track_id", "x1", "y1", "x2", "y2"}
    missing = required - set(tracks_df.columns)
    if missing:
        raise ValueError(f"tracks_df missing columns: {missing}")

    rows = []
    for tid, g in tracks_df.groupby("track_id", sort=False):
        if len(g) < thr.min_track_frames:
            rows.append({"track_id": tid, "in_roadway_any": False, "direction": "unknown", "cos_to_flow": None, "in_right_band_any": False, "right_side_violation": False})
            continue

        labels = label_track_frames(g, roi, thr)

        # roadway_any: count roadway OR crosswalk (since crosswalk is roadway semantics)
        in_road_any = ("roadway" in labels) or ("crosswalk" in labels)
        in_band_any = (thr.band_label in labels)

        direction, cos = direction_cosine(g, roi, min_move_px=thr.min_move_px)

        # only evaluate along_flow population; against_flow shouldn't be counted here
        right_side_violation = bool(direction == "along_flow" and in_road_any and (not in_band_any))

        rows.append({
            "track_id": tid,
            "in_roadway_any": bool(in_road_any),
            "direction": direction,
            "cos_to_flow": cos,
            "in_right_band_any": bool(in_band_any),
            "right_side_violation": right_side_violation,
        })

    events_df = pd.DataFrame(rows)
    denom_mask = (events_df["direction"] == "along_flow") & (events_df["in_roadway_any"])
    denom = int(denom_mask.sum()) if len(events_df) else 0
    num = int(events_df.loc[denom_mask, "right_side_violation"].sum()) if denom > 0 else 0

    summary = {
        "N_tracks": int(events_df.shape[0]),
        "N_along_flow_on_roadway": denom,
        "N_right_side_violations": num,
        "share_right_side_violations": (num / denom) if denom > 0 else None,
    }
    return events_df, summary

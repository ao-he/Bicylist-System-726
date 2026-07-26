# src/inference/wrong_way.py
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd

from src.scene.roi import ROIMaskEngine
from src.inference.common import FrameLabelThresholds, label_track_frames, direction_cosine

@dataclass
class WrongWayThresholds(FrameLabelThresholds):
    min_track_frames: int = 5
    min_move_px: float = 8.0

def infer_wrong_way_per_track(
    tracks_df: pd.DataFrame,
    roi: ROIMaskEngine,
    thr: WrongWayThresholds = WrongWayThresholds(),
) -> Tuple[pd.DataFrame, Dict]:
    required = {"track_id", "x1", "y1", "x2", "y2"}
    missing = required - set(tracks_df.columns)
    if missing:
        raise ValueError(f"tracks_df missing columns: {missing}")

    rows = []
    for tid, g in tracks_df.groupby("track_id", sort=False):
        if len(g) < thr.min_track_frames:
            rows.append({"track_id": tid, "in_bike_lane_any": False, "direction": "unknown", "cos_to_flow": None, "wrong_way": False})
            continue

        labels = label_track_frames(g, roi, thr)
        in_bike_any = ("bike_lane" in labels)

        direction, cos = direction_cosine(g, roi, min_move_px=thr.min_move_px)
        wrong_way = bool(in_bike_any and direction == "against_flow")

        rows.append({
            "track_id": tid,
            "in_bike_lane_any": bool(in_bike_any),
            "direction": direction,
            "cos_to_flow": cos,
            "wrong_way": wrong_way,
        })

    events_df = pd.DataFrame(rows)
    summary = {
        "N_tracks": int(events_df.shape[0]),
        "N_in_bike_lane": int(events_df["in_bike_lane_any"].sum()) if len(events_df) else 0,
        "N_wrong_way": int(events_df["wrong_way"].sum()) if len(events_df) else 0,
    }
    return events_df, summary

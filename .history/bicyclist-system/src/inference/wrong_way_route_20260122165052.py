# src/inference/wrong_way_route.py
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd

from src.scene.roi import ROIMaskEngine
from src.inference.common_route import FrameLabelThresholds, label_track_frames, direction_cosine

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

    has_bike = bool(roi.cfg.rois.get("bike_lane")) and len(roi.cfg.rois["bike_lane"]) > 0
    flow = roi.flow_vector()
    scene_applicable = has_bike and (flow is not None)

    rows = []
    for tid, g in tracks_df.groupby("track_id", sort=False):
        if (not scene_applicable) or (len(g) < thr.min_track_frames):
            rows.append({
                "track_id": tid,
                "in_bike_lane_any": pd.NA,
                "direction": pd.NA,
                "cos_to_flow": pd.NA,
                "wrong_way": pd.NA,
            })
            continue

        labels = label_track_frames(g, roi, thr)
        in_bike_any = ("bike_lane" in labels)

        direction, cos = direction_cosine(g, roi, min_move_px=thr.min_move_px)
        if direction == "unknown" or cos is None:
            rows.append({
                "track_id": tid,
                "in_bike_lane_any": bool(in_bike_any),
                "direction": pd.NA,
                "cos_to_flow": pd.NA,
                "wrong_way": pd.NA,
            })
            continue

        wrong_way = bool(in_bike_any and direction == "against_flow")

        rows.append({
            "track_id": tid,
            "in_bike_lane_any": bool(in_bike_any),
            "direction": direction,
            "cos_to_flow": float(cos),
            "wrong_way": wrong_way,
        })

    events_df = pd.DataFrame(rows)
    applicable_mask = events_df["wrong_way"].notna()
    denom = int(applicable_mask.sum()) if len(events_df) else 0
    num = int((events_df.loc[applicable_mask, "wrong_way"] == True).sum()) if denom > 0 else 0

    summary = {
        "N_tracks": int(events_df.shape[0]),
        "N_applicable": denom,
        "N_wrong_way": num,
        "share_wrong_way": (num / denom) if denom > 0 else None,
        "scene_applicable": bool(scene_applicable),
    }
    return events_df, summary

# src/inference/crossing.py
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd

from src.scene.roi import ROIMaskEngine
from src.inference.common import FrameLabelThresholds, label_track_frames, longest_run

@dataclass
class CrossingThresholds(FrameLabelThresholds):
    min_track_frames: int = 5
    min_consecutive_frames: int = 3

def infer_crossing_per_track(
    tracks_df: pd.DataFrame,
    roi: ROIMaskEngine,
    thr: CrossingThresholds = CrossingThresholds(),
) -> Tuple[pd.DataFrame, Dict]:
    required = {"track_id", "x1", "y1", "x2", "y2"}
    missing = required - set(tracks_df.columns)
    if missing:
        raise ValueError(f"tracks_df missing columns: {missing}")

    rows = []
    for tid, g in tracks_df.groupby("track_id", sort=False):
        if len(g) < thr.min_track_frames:
            rows.append({"track_id": tid, "crossing_any": False, "crossing_crosswalk": False, "crossing_non_crosswalk": False})
            continue

        labels = label_track_frames(g, roi, thr)

        cw_run = longest_run(labels, "crosswalk")
        rd_run = longest_run(labels, "roadway")

        crossing_crosswalk = cw_run >= thr.min_consecutive_frames
        # non-crosswalk only if NO crosswalk crossing
        crossing_non_crosswalk = (not crossing_crosswalk) and (rd_run >= thr.min_consecutive_frames)
        crossing_any = crossing_crosswalk or crossing_non_crosswalk

        rows.append({
            "track_id": tid,
            "crossing_any": bool(crossing_any),
            "crossing_crosswalk": bool(crossing_crosswalk),
            "crossing_non_crosswalk": bool(crossing_non_crosswalk),
        })

    events_df = pd.DataFrame(rows)
    summary = {
        "N_tracks": int(events_df.shape[0]),
        "N_crossing_any": int(events_df["crossing_any"].sum()) if len(events_df) else 0,
        "N_crosswalk": int(events_df["crossing_crosswalk"].sum()) if len(events_df) else 0,
        "N_non_crosswalk": int(events_df["crossing_non_crosswalk"].sum()) if len(events_df) else 0,
    }
    return events_df, summary

# src/inference/wrong_way.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd

from src.scene.roi import ROIMaskEngine
from src.inference.common import FrameLabelThresholds, label_track_frames, direction_cosine


@dataclass
class WrongWayThresholds(FrameLabelThresholds):
    min_track_frames: int = 5
    min_move_px: float = 8.0  # too small => direction unknown


def infer_wrong_way_per_track(
    tracks_df: pd.DataFrame,
    roi: ROIMaskEngine,
    thr: WrongWayThresholds = WrongWayThresholds(),
) -> Tuple[pd.DataFrame, Dict]:
    """
    Route A (recommended):
      - Only infer wrong-way when the scene has flow semantics (roi.flow_vector() != None)
        AND has bike_lane ROI.
      - Otherwise mark as N/A (pd.NA) so it won't pollute rates.

    Output per track:
      track_id, in_bike_lane_any, direction, cos_to_flow, wrong_way
      where fields can be pd.NA when not applicable.
    """
    required = {"track_id", "x1", "y1", "x2", "y2"}
    missing = required - set(tracks_df.columns)
    if missing:
        raise ValueError(f"tracks_df missing columns: {missing}")

    # scene semantics availability
    has_bike = bool(roi.cfg.rois.get("bike_lane")) and len(roi.cfg.rois["bike_lane"]) > 0
    flow = roi.flow_vector()
    scene_applicable =  (flow is not None)

    rows = []
    for tid, g in tracks_df.groupby("track_id", sort=False):
        # If scene can't support wrong-way OR track too short => N/A
        if (not scene_applicable) or (len(g) < thr.min_track_frames):
            rows.append({
                "track_id": tid,
                "in_bike_lane_any": pd.NA,
                "direction": pd.NA,
                "cos_to_flow": pd.NA,
                "wrong_way": pd.NA,
            })
            continue

        # frame labels (shared consistent logic)
        labels = label_track_frames(g, roi, thr)
        in_bike_any = ("bike_lane" in labels)

        # direction vs flow (shared consistent logic)
        direction, cos = direction_cosine(g, roi, min_move_px=thr.min_move_px)

        # If direction cannot be determined (too small movement), mark as N/A
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

    # Summary: only compute rates on applicable tracks (wrong_way not NA)
    if len(events_df) == 0:
        summary = {"N_tracks": 0, "N_applicable": 0, "N_wrong_way": 0, "share_wrong_way": None}
        return events_df, summary

    applicable_mask = events_df["wrong_way"].notna()
    denom = int(applicable_mask.sum())
    num = int((events_df.loc[applicable_mask, "wrong_way"] == True).sum()) if denom > 0 else 0

    summary = {
        "N_tracks": int(events_df.shape[0]),
        "N_applicable": denom,
        "N_wrong_way": num,
        "share_wrong_way": (num / denom) if denom > 0 else None,
        # optional quick debug:
        "scene_applicable": bool(scene_applicable),
    }
    return events_df, summary
    #
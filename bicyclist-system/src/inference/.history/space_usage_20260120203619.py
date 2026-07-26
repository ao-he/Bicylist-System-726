# src/inference/space_usage.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from src.scene.roi import ROIMaskEngine, BBox


@dataclass
class SpaceUsageThresholds:
    T_ROAD: float = 0.30
    T_BIKE: float = 0.25
    T_SIDE: float = 0.30
    shrink_frac: float = 0.08


def _row_bbox(r) -> BBox:
    return (float(r.x1), float(r.y1), float(r.x2), float(r.y2))


def infer_space_usage_per_track(
    tracks_df: pd.DataFrame,
    roi: ROIMaskEngine,
    thr: SpaceUsageThresholds = SpaceUsageThresholds(),
) -> Tuple[pd.DataFrame, Dict]:
    """
    Space usage inference per track.

    Output per track:
      - in_roadway_any / in_bike_lane_any / in_sidewalk_any (ever true)
      - dominant_space: one of {"bike_lane","sidewalk","roadway","unknown"}
        determined by which ROI achieves the highest max overlap across frames
        with a simple priority tie-break: bike_lane > sidewalk > roadway
    """
    required = {"track_id", "x1", "y1", "x2", "y2"}
    missing = required - set(tracks_df.columns)
    if missing:
        raise ValueError(f"tracks_df missing columns: {missing}")

    has_road = roi.cfg.rois.get("roadway") and len(roi.cfg.rois["roadway"]) > 0
    has_bike = roi.cfg.rois.get("bike_lane") and len(roi.cfg.rois["bike_lane"]) > 0
    has_side = roi.cfg.rois.get("sidewalk") and len(roi.cfg.rois["sidewalk"]) > 0

    rows = []
    for tid, g in tracks_df.groupby("track_id", sort=False):
        bboxes = g.apply(_row_bbox, axis=1).tolist()

        max_road = 0.0
        max_bike = 0.0
        max_side = 0.0

        in_road_any = False
        in_bike_any = False
        in_side_any = False

        for bb in bboxes:
            if has_road:
                ov = roi.overlap_ratio(bb, "roadway", shrink=thr.shrink_frac)
                max_road = max(max_road, ov)
                if ov > thr.T_ROAD:
                    in_road_any = True

            if has_bike:
                ov = roi.overlap_ratio(bb, "bike_lane", shrink=thr.shrink_frac)
                max_bike = max(max_bike, ov)
                if ov > thr.T_BIKE:
                    in_bike_any = True

            if has_side:
                ov = roi.overlap_ratio(bb, "sidewalk", shrink=thr.shrink_frac)
                max_side = max(max_side, ov)
                if ov > thr.T_SIDE:
                    in_side_any = True

        # dominant space based on max overlap with tie-break priority
        dominant = "unknown"
        best = 0.0
        # priority: bike > sidewalk > roadway
        if max_bike > best and max_bike > thr.T_BIKE:
            dominant, best = "bike_lane", max_bike
        if max_side > best and max_side > thr.T_SIDE:
            dominant, best = "sidewalk", max_side
        if max_road > best and max_road > thr.T_ROAD:
            dominant, best = "roadway", max_road

        rows.append({
            "track_id": tid,
            "in_bike_lane_any": in_bike_any,
            "in_sidewalk_any": in_side_any,
            "in_roadway_any": in_road_any,
            "max_bike_ov": float(max_bike),
            "max_side_ov": float(max_side),
            "max_road_ov": float(max_road),
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

# src/inference/crossing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from src.scene.roi import ROIMaskEngine, BBox


@dataclass
class CrossingThresholds:
    # overlap thresholds
    T_ROAD: float = 0.30
    T_CW: float = 0.25
    # optional bbox shrink to reduce boundary noise
    shrink_frac: float = 0.08


def _row_bbox(r) -> BBox:
    return (float(r.x1), float(r.y1), float(r.x2), float(r.y2))


def infer_crossing_per_track(
    tracks_df: pd.DataFrame,
    roi: ROIMaskEngine,
    thr: CrossingThresholds = CrossingThresholds(),
) -> Tuple[pd.DataFrame, Dict]:
    """
    Infer crossing events per track for locations with roadway/crosswalk semantics.

    Requires tracks_df columns: track_id, x1,y1,x2,y2
    frame is optional.

    Definitions:
      crossing_any: ever overlaps roadway above T_ROAD
      crossing_crosswalk: ever overlaps crosswalk above T_CW AND bottom-center in crosswalk
      crossing_non_crosswalk = crossing_any and not crossing_crosswalk
    """
    required = {"track_id", "x1", "y1", "x2", "y2"}
    missing = required - set(tracks_df.columns)
    if missing:
        raise ValueError(f"tracks_df missing columns: {missing}")

    # If the location has no roadway polygon, we cannot infer crossing_any
    has_roadway = roi.cfg.rois.get("roadway") and len(roi.cfg.rois["roadway"]) > 0
    has_crosswalk = roi.cfg.rois.get("crosswalk") and len(roi.cfg.rois["crosswalk"]) > 0

    rows = []
    for tid, g in tracks_df.groupby("track_id", sort=False):
        # per-frame features
        bboxes = g.apply(_row_bbox, axis=1).tolist()

        road_ov = []
        cw_ov = []
        bc_in_cw = []

        for bb in bboxes:
            if has_roadway:
                road_ov.append(roi.overlap_ratio(bb, "roadway", shrink=thr.shrink_frac))
            else:
                road_ov.append(0.0)

            if has_crosswalk:
                cw_ov.append(roi.overlap_ratio(bb, "crosswalk", shrink=thr.shrink_frac))
                bc = roi.bbox_bottom_center(bb)
                bc_in_cw.append(roi.point_in_roi(bc, "crosswalk"))
            else:
                cw_ov.append(0.0)
                bc_in_cw.append(False)

        crossing_any = bool(any(v > thr.T_ROAD for v in road_ov)) if has_roadway else False
        crossing_crosswalk = bool(any((ov > thr.T_CW) and inside for ov, inside in zip(cw_ov, bc_in_cw))) if has_crosswalk else False
        crossing_non_crosswalk = bool(crossing_any and (not crossing_crosswalk))

        rows.append({
            "track_id": tid,
            "crossing_any": crossing_any,
            "crossing_crosswalk": crossing_crosswalk,
            "crossing_non_crosswalk": crossing_non_crosswalk,
        })

    events_df = pd.DataFrame(rows)

    summary = {
        "N_tracks": int(events_df.shape[0]),
        "N_crossing_any": int(events_df["crossing_any"].sum()) if len(events_df) else 0,
        "N_crosswalk": int(events_df["crossing_crosswalk"].sum()) if len(events_df) else 0,
        "N_non_crosswalk": int(events_df["crossing_non_crosswalk"].sum()) if len(events_df) else 0,
    }
    return events_df, summary

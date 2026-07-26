# src/inference/right_side.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from src.scene.roi import ROIMaskEngine, BBox


@dataclass
class RightSideThresholds:
    """
    Right-side compliance thresholds.

    Interpretation:
      - roadway: the analysis corridor / shared road
      - bike_lane: the "right-side band" for the target direction (along_flow)
      - flow arrow: defines the target direction as along_flow
    """
    T_ROAD: float = 0.30
    T_BAND: float = 0.25
    shrink_frac: float = 0.08
    min_move_px: float = 8.0


def _row_bbox(r) -> BBox:
    return (float(r.x1), float(r.y1), float(r.x2), float(r.y2))


def _unit(vx: float, vy: float) -> Tuple[float, float, float]:
    n = (vx * vx + vy * vy) ** 0.5
    if n < 1e-6:
        return 0.0, 0.0, 0.0
    return vx / n, vy / n, n


def infer_right_side_per_track(
    tracks_df: pd.DataFrame,
    roi: ROIMaskEngine,
    thr: RightSideThresholds = RightSideThresholds(),
) -> Tuple[pd.DataFrame, Dict]:
    """
    Right-side compliance inference (shared-road / corridor scenario).

    Requires:
      - roadway polygons exist (analysis corridor)
      - bike_lane polygons exist (used as "right-side band" for along_flow)
      - flow arrow exists (roi.flow_vector() != None)

    Logic per track:
      1) in_roadway_any: ever overlaps roadway > T_ROAD   (no hard point-in needed)
      2) in_band_any:    ever overlaps bike_lane > T_BAND AND bottom-center in bike_lane
      3) direction:
         v = last_bc - first_bc
         cos = unit(v) · flow
         cos >= 0 => along_flow
         cos < 0  => against_flow
      4) right_side_violation:
         direction == along_flow AND in_roadway_any AND (not in_band_any)

    Notes:
      - We only evaluate right-side compliance for along_flow population.
      - against_flow population is not counted as right-side violation here.
    """
    required = {"track_id", "x1", "y1", "x2", "y2"}
    missing = required - set(tracks_df.columns)
    if missing:
        raise ValueError(f"tracks_df missing columns: {missing}")

    has_road = roi.cfg.rois.get("roadway") and len(roi.cfg.rois["roadway"]) > 0
    has_band = roi.cfg.rois.get("bike_lane") and len(roi.cfg.rois["bike_lane"]) > 0
    flow = roi.flow_vector()

    rows = []
    for tid, g in tracks_df.groupby("track_id", sort=False):
        bboxes = g.apply(_row_bbox, axis=1).tolist()

        # If missing semantics, return safe defaults
        if (not has_road) or (not has_band) or (flow is None):
            rows.append({
                "track_id": tid,
                "in_roadway_any": False,
                "direction": "unknown",
                "cos_to_flow": None,
                "in_right_band_any": False,
                "right_side_violation": False,
            })
            continue

        # Per-frame membership
        in_road_any = False
        in_band_any = False
        bcs = []

        for bb in bboxes:
            road_ov = roi.overlap_ratio(bb, "roadway", shrink=thr.shrink_frac)
            if road_ov > thr.T_ROAD:
                in_road_any = True

            band_ov = roi.overlap_ratio(bb, "bike_lane", shrink=thr.shrink_frac)
            bc = roi.bbox_bottom_center(bb)
            bcs.append(bc)
            if (band_ov > thr.T_BAND) and roi.point_in_roi(bc, "bike_lane"):
                in_band_any = True

        # Direction based on first/last bottom-center
        x0, y0 = bcs[0]
        x1, y1 = bcs[-1]
        vx, vy = (x1 - x0), (y1 - y0)
        ux, uy, mag = _unit(vx, vy)

        if mag < thr.min_move_px:
            direction = "unknown"
            cos = None
        else:
            cos = float(ux * flow[0] + uy * flow[1])
            direction = "along_flow" if cos >= 0 else "against_flow"

        right_side_violation = bool(direction == "along_flow" and in_road_any and (not in_band_any))

        rows.append({
            "track_id": tid,
            "in_roadway_any": in_road_any,
            "direction": direction,
            "cos_to_flow": cos,
            "in_right_band_any": in_band_any,
            "right_side_violation": right_side_violation,
        })

    events_df = pd.DataFrame(rows)

    # Summary: only evaluate rate among along_flow & in_roadway tracks
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

# src/inference/wrong_way.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from src.scene.roi import ROIMaskEngine, BBox


@dataclass
class WrongWayThresholds:
    T_BIKE: float = 0.25
    shrink_frac: float = 0.08
    min_move_px: float = 8.0  # too small => direction unknown


def _row_bbox(r) -> BBox:
    return (float(r.x1), float(r.y1), float(r.x2), float(r.y2))


def _unit(vx: float, vy: float) -> Tuple[float, float, float]:
    n = (vx * vx + vy * vy) ** 0.5
    if n < 1e-6:
        return 0.0, 0.0, 0.0
    return vx / n, vy / n, n


def infer_wrong_way_per_track(
    tracks_df: pd.DataFrame,
    roi: ROIMaskEngine,
    thr: WrongWayThresholds = WrongWayThresholds(),
) -> Tuple[pd.DataFrame, Dict]:
    """
    Infer wrong-way riding per track, only within bike_lane ROI.

    Requires:
      - bike_lane polygons exist
      - flow arrow exists in JSON (roi.flow_vector() != None)

    Logic:
      in_bike_lane_any = exists frame with bike overlap > T_BIKE AND bottom-center in bike_lane
      direction:
        v = last_bc - first_bc
        cos = unit(v) · flow
        cos >= 0 => along_flow
        cos < 0  => against_flow
      wrong_way = in_bike_lane_any AND direction == against_flow
    """
    required = {"track_id", "x1", "y1", "x2", "y2"}
    missing = required - set(tracks_df.columns)
    if missing:
        raise ValueError(f"tracks_df missing columns: {missing}")

    has_bike = roi.cfg.rois.get("bike_lane") and len(roi.cfg.rois["bike_lane"]) > 0
    flow = roi.flow_vector()

    # If missing required scene semantics, return empty flags (still consistent)
    rows = []
    for tid, g in tracks_df.groupby("track_id", sort=False):
        bboxes = g.apply(_row_bbox, axis=1).tolist()

        if not has_bike or flow is None:
            rows.append({
                "track_id": tid,
                "in_bike_lane_any": False,
                "direction": "unknown",
                "cos_to_flow": None,
                "wrong_way": False,
            })
            continue

        # compute in_bike_lane_any
        in_bike_any = False
        bcs = []
        for bb in bboxes:
            bike_ov = roi.overlap_ratio(bb, "bike_lane", shrink=thr.shrink_frac)
            bc = roi.bbox_bottom_center(bb)
            bcs.append(bc)
            if (bike_ov > thr.T_BIKE) and roi.point_in_roi(bc, "bike_lane"):
                in_bike_any = True

        # direction based on first/last bottom-center
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

        wrong_way = bool(in_bike_any and direction == "against_flow")

        rows.append({
            "track_id": tid,
            "in_bike_lane_any": in_bike_any,
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

# src/inference/common_route.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd

from src.scene.roi import ROIMaskEngine, BBox


@dataclass
class FrameLabelThresholds:
    T_ROAD: float = 0.30
    T_BIKE: float = 0.25
    T_SIDE: float = 0.30
    shrink_frac: float = 0.08
    use_bottom_center_gate: bool = True


def _row_bbox(r) -> BBox:
    return (float(r.x1), float(r.y1), float(r.x2), float(r.y2))


def frame_space_label(bb: BBox, roi: ROIMaskEngine, thr: FrameLabelThresholds) -> str:
    """
    Route-only frame labeling.

    Priority:
      crosswalk > bike_lane > sidewalk > roadway > unknown

    Crosswalk:
      bottom-edge 3-point sampling with point-in-ROI (robust for narrow polygons).
    """
    x1, y1, x2, y2 = bb
    bc = roi.bbox_bottom_center(bb)

    def has_roi(name: str) -> bool:
        return bool(roi.cfg.rois.get(name)) and len(roi.cfg.rois[name]) > 0

    # 1) crosswalk: always define pts (fix UnboundLocalError)
    if has_roi("crosswalk"):
        xs = [x1, (x1 + x2) / 2.0, x2]
        pts = [(float(x), float(y2)) for x in xs]
        if any(roi.point_in_roi(p, "crosswalk") for p in pts):
            return "crosswalk"

    # 2) bike lane: overlap + optional bottom-center gate
    if has_roi("bike_lane"):
        ov = roi.overlap_ratio(bb, "bike_lane", shrink=thr.shrink_frac)
        if ov > thr.T_BIKE:
            if (not thr.use_bottom_center_gate) or roi.point_in_roi(bc, "bike_lane"):
                return "bike_lane"

    # 3) sidewalk: overlap + optional bottom-center gate
    if has_roi("sidewalk"):
        ov = roi.overlap_ratio(bb, "sidewalk", shrink=thr.shrink_frac)
        if ov > thr.T_SIDE:
            if (not thr.use_bottom_center_gate) or roi.point_in_roi(bc, "sidewalk"):
                return "sidewalk"

    # 4) roadway: overlap only
    if has_roi("roadway"):
        ov = roi.overlap_ratio(bb, "roadway", shrink=thr.shrink_frac)
        if ov > thr.T_ROAD:
            return "roadway"

    return "unknown"


def label_track_frames(track_df: pd.DataFrame, roi: ROIMaskEngine, thr: FrameLabelThresholds) -> List[str]:
    bbs = track_df.apply(_row_bbox, axis=1).tolist()
    return [frame_space_label(bb, roi, thr) for bb in bbs]


def longest_run(labels: List[str], target: str) -> int:
    best = 0
    cur = 0
    for x in labels:
        if x == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def direction_cosine(track_df: pd.DataFrame, roi: ROIMaskEngine, min_move_px: float = 8.0) -> Tuple[str, Optional[float]]:
    flow = roi.flow_vector()
    if flow is None or len(track_df) == 0:
        return "unknown", None

    bbs = track_df.apply(_row_bbox, axis=1).tolist()
    bcs = [roi.bbox_bottom_center(bb) for bb in bbs]

    x0, y0 = bcs[0]
    x1, y1 = bcs[-1]
    vx, vy = (x1 - x0), (y1 - y0)

    mag = (vx * vx + vy * vy) ** 0.5
    if mag < min_move_px:
        return "unknown", None

    ux, uy = vx / mag, vy / mag
    cos = float(ux * flow[0] + uy * flow[1])

    direction = "along_flow" if cos >= 0 else "against_flow"
    return direction, cos

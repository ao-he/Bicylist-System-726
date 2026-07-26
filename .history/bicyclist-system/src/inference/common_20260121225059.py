# src/inference/common.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd

from src.scene.roi import ROIMaskEngine, BBox


@dataclass
class FrameLabelThresholds:
    """
    Shared thresholds for frame-level space labeling.

    IMPORTANT (Route A design):
      - Crosswalk should be captured robustly even when bbox overlap is small.
      - Therefore, crosswalk uses a POINT-IN-ROI rule (bottom-center) by default.
    """
    T_ROAD: float = 0.30
    T_BIKE: float = 0.25
    T_SIDE: float = 0.30
    T_CW: float = 0.25  # kept for fallback/debug; not used in crosswalk point-rule
    shrink_frac: float = 0.08
    use_bottom_center_gate: bool = True  # used for bike_lane/sidewalk; crosswalk uses point-rule regardless


def _row_bbox(r) -> BBox:
    """Convert a dataframe row into a BBox tuple."""
    return (float(r.x1), float(r.y1), float(r.x2), float(r.y2))


def frame_space_label(
    bb: BBox,
    roi: ROIMaskEngine,
    thr: FrameLabelThresholds
) -> str:
    """
    Assign ONE spatial label to a bbox for a single frame.

    Priority (high -> low):
      crosswalk > bike_lane > sidewalk > roadway > unknown

    Crosswalk rule (robust):
      - Use bottom-center point-in-ROI ONLY (no overlap threshold),
        because crosswalk is often narrow and bbox overlap can be small.

    Other rules:
      - bike_lane/sidewalk: overlap threshold + optional bottom-center gate
      - roadway: overlap threshold
    """
    bc = roi.bbox_bottom_center(bb)

    def has_roi(name: str) -> bool:
        return bool(roi.cfg.rois.get(name)) and len(roi.cfg.rois[name]) > 0

    # --- 1) Crosswalk: point-in-ROI only (no overlap threshold) ---
    if has_roi("crosswalk"):
        x1, y1, x2, y2 = bb
        xs = [x1, (x1+x2)/2, x2]
        pts = [(float(x), float(y2)) for x in xs]

    if any(roi.point_in_roi(p, "crosswalk") for p in pts):
        return "crosswalk"

    # --- 2) Bike lane: overlap + (optional) bottom-center gate ---
    if has_roi("bike_lane"):
        ov = roi.overlap_ratio(bb, "bike_lane", shrink=thr.shrink_frac)
        if ov > thr.T_BIKE:
            if (not thr.use_bottom_center_gate) or roi.point_in_roi(bc, "bike_lane"):
                return "bike_lane"

    # --- 3) Sidewalk: overlap + (optional) bottom-center gate ---
    if has_roi("sidewalk"):
        ov = roi.overlap_ratio(bb, "sidewalk", shrink=thr.shrink_frac)
        if ov > thr.T_SIDE:
            if (not thr.use_bottom_center_gate) or roi.point_in_roi(bc, "sidewalk"):
                return "sidewalk"

    # --- 4) Roadway: overlap only ---
    if has_roi("roadway"):
        ov = roi.overlap_ratio(bb, "roadway", shrink=thr.shrink_frac)
        if ov > thr.T_ROAD:
            return "roadway"

    return "unknown"


def label_track_frames(
    track_df: pd.DataFrame,
    roi: ROIMaskEngine,
    thr: FrameLabelThresholds
) -> List[str]:
    """
    Label all frames of a track using the unified frame_space_label().
    """
    bbs = track_df.apply(_row_bbox, axis=1).tolist()
    return [frame_space_label(bb, roi, thr) for bb in bbs]


def longest_run(labels: List[str], target: str) -> int:
    """
    Compute the longest consecutive run of a given label in a sequence.
    Used for crossing detection (min_consecutive_frames logic).
    """
    best = 0
    cur = 0
    for x in labels:
        if x == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def direction_cosine(
    track_df: pd.DataFrame,
    roi: ROIMaskEngine,
    min_move_px: float = 8.0
) -> Tuple[str, Optional[float]]:
    """
    Compute movement direction of a track relative to the scene flow vector.

    Returns:
      direction: {"along_flow", "against_flow", "unknown"}
      cos_to_flow: cosine similarity (None if direction is unknown)

    Rules:
      - If roi.flow_vector() is None -> ("unknown", None)
      - If displacement magnitude < min_move_px -> ("unknown", None)
      - Uses bottom-center of first and last bbox
    """
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
#
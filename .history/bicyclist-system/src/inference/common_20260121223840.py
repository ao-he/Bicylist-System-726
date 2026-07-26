# src/inference/common.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd

from src.scene.roi import ROIMaskEngine, BBox


# ---------------------------------------------------------------------
# Thresholds shared by all inference modules
# ---------------------------------------------------------------------
@dataclass
class FrameLabelThresholds:
    """
    Shared thresholds for frame-level space labeling.

    All inference modules (crossing / wrong-way / etc.) must rely on this
    class to guarantee rule consistency.
    """
    T_ROAD: float = 0.30
    T_BIKE: float = 0.25
    T_SIDE: float = 0.30
    T_CW: float = 0.1
    shrink_frac: float = 0.08
    use_bottom_center_gate: bool = True


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _row_bbox(r) -> BBox:
    """Convert a dataframe row into a BBox tuple."""
    return (float(r.x1), float(r.y1), float(r.x2), float(r.y2))


# ---------------------------------------------------------------------
# Frame-level space labeling (THE SINGLE SOURCE OF TRUTH)
# ---------------------------------------------------------------------
def frame_space_label(
    bb: BBox,
    roi: ROIMaskEngine,
    thr: FrameLabelThresholds
) -> str:
    """
    Assign ONE spatial label to a bbox for a single frame.

    Priority (high -> low):
      crosswalk > bike_lane > sidewalk > roadway > unknown

    This priority is critical to avoid 'crosswalk being overridden by roadway'.
    """
    bc = roi.bbox_bottom_center(bb) if thr.use_bottom_center_gate else None

    def ok(name: str, T: float) -> bool:
        if not (roi.cfg.rois.get(name) and len(roi.cfg.rois[name]) > 0):
            return False
        ov = roi.overlap_ratio(bb, name, shrink=thr.shrink_frac)
        if ov <= T:
            return False
        if thr.use_bottom_center_gate and name in { "bike_lane", "sidewalk"}:
            return roi.point_in_roi(bc, name)
        return True

    # IMPORTANT: priority order is intentional and must not change
    if ok("crosswalk", thr.T_CW):
        return "crosswalk"
    if ok("bike_lane", thr.T_BIKE):
        return "bike_lane"
    if ok("sidewalk", thr.T_SIDE):
        return "sidewalk"
    if ok("roadway", thr.T_ROAD):
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


# ---------------------------------------------------------------------
# Track-level utilities
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Direction inference (used by wrong-way, optional capabilities)
# ---------------------------------------------------------------------
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

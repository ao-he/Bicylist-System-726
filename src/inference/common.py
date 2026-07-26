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

    Design:
      - Crosswalk is captured robustly using a bottom-edge point-in-ROI rule.
      - For bike_lane / sidewalk / roadway, we use bottom-edge multi-point voting
        (hard assignment) to reduce perspective/bbox bias.

    Notes:
      - T_* overlap thresholds are kept for backward compatibility, but are NOT used
        by the new voting-based labeling for bike/side/road.
    """
    # kept for compatibility (no longer used for bike/side/road in voting mode)
    T_ROAD: float = 0.30
    T_BIKE: float = 0.25
    T_SIDE: float = 0.30
    T_CW: float = 0.25
    shrink_frac: float = 0.08
    use_bottom_center_gate: bool = True  # not used in voting mode

    # NEW: bottom-edge voting controls
    bottom_edge_npts: int = 5   # sample points along bbox bottom edge
    vote_min_pts: int = 2       # require >= this many points inside ROI to accept label


def _row_bbox(r) -> BBox:
    """Convert a dataframe row into a BBox tuple."""
    return (float(r.x1), float(r.y1), float(r.x2), float(r.y2))


def _bottom_edge_points(bb: BBox, npts: int) -> List[Tuple[float, float]]:
    """Sample npts points uniformly along bbox bottom edge."""
    x1, y1, x2, y2 = bb
    if npts <= 1:
        xs = [(x1 + x2) / 2.0]
    else:
        w = x2 - x1
        xs = [x1 + (i / (npts - 1)) * w for i in range(npts)]
    return [(float(x), float(y2)) for x in xs]


def frame_space_label(
    bb: BBox,
    roi: ROIMaskEngine,
    thr: FrameLabelThresholds
) -> str:
    """
    Assign ONE spatial label to a bbox for a single frame.

    Priority (high -> low):
      crosswalk > bike_lane > sidewalk > roadway > unknown

    Ignore-zone:
      - If ignore_zone exists and bbox bottom-center is inside ignore_zone -> unknown.

    Crosswalk rule (robust):
      - Use bottom-edge point-in-ROI ONLY (no overlap threshold).

    Bike_lane/Sidewalk/Roadway rule (hard assignment):
      - Use bottom-edge multi-point voting.
      - Count how many bottom-edge points fall inside each ROI.
      - Pick the ROI with the largest count, but require >= vote_min_pts.
      - If no ROI reaches vote_min_pts -> unknown.
    """
    bc = roi.bbox_bottom_center(bb)

    def has_roi(name: str) -> bool:
        return bool(roi.cfg.rois.get(name)) and len(roi.cfg.rois[name]) > 0

    # 0) Ignore-zone
    if has_roi("ignore_zone") and roi.point_in_roi(bc, "ignore_zone"):
        return "unknown"

    # Prepare bottom-edge points
    pts = _bottom_edge_points(bb, int(thr.bottom_edge_npts))

    # 1) Crosswalk: any-point
    if has_roi("crosswalk"):
        if any(roi.point_in_roi(p, "crosswalk") for p in pts):
            return "crosswalk"

    # 2-4) Vote among bike_lane/sidewalk/roadway
    vote_names = ["bike_lane", "sidewalk", "roadway"]
    votes = {rn: 0 for rn in vote_names}
    for rn in vote_names:
        if has_roi(rn):
            votes[rn] = sum(1 for p in pts if roi.point_in_roi(p, rn))
        else:
            votes[rn] = 0

    best = max(votes.keys(), key=lambda k: votes[k])
    if votes[best] >= int(thr.vote_min_pts):
        return best

    return "unknown"


def label_track_frames(
    track_df: pd.DataFrame,
    roi: ROIMaskEngine,
    thr: FrameLabelThresholds
) -> List[str]:
    """Label all frames of a track using the unified frame_space_label()."""
    bbs = track_df.apply(_row_bbox, axis=1).tolist()
    return [frame_space_label(bb, roi, thr) for bb in bbs]


def longest_run(labels: List[str], target: str) -> int:
    """Compute the longest consecutive run of a given label in a sequence."""
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
      cos_to_flow: cosine similarity (None if unknown)

    Robust version:
      - Uses bottom-center points across the whole track.
      - Uses median per-step displacement to reduce jitter / bbox noise.
      - Sorts by frame column if present.
    """
    flow = roi.flow_vector()
    if flow is None or len(track_df) < 2:
        return "unknown", None

    g = track_df
    if "frame" in g.columns:
        g = g.sort_values("frame")
    elif "frame_global" in g.columns:
        g = g.sort_values("frame_global")

    bbs = g.apply(_row_bbox, axis=1).tolist()
    bcs = [roi.bbox_bottom_center(bb) for bb in bbs]
    if len(bcs) < 2:
        return "unknown", None

    dxs = [bcs[i + 1][0] - bcs[i][0] for i in range(len(bcs) - 1)]
    dys = [bcs[i + 1][1] - bcs[i][1] for i in range(len(bcs) - 1)]

    vx = float(pd.Series(dxs).median())
    vy = float(pd.Series(dys).median())

    mag = (vx * vx + vy * vy) ** 0.5
    if mag < min_move_px:
        return "unknown", None

    ux, uy = vx / mag, vy / mag

    fx, fy = float(flow[0]), float(flow[1])
    fmag = (fx * fx + fy * fy) ** 0.5
    if fmag < 1e-9:
        return "unknown", None
    fx, fy = fx / fmag, fy / fmag

    cos = float(ux * fx + uy * fy)
    direction = "along_flow" if cos >= 0 else "against_flow"
    return direction, cos

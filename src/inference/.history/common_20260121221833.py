# src/inference/common.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import pandas as pd

from src.scene.roi import ROIMaskEngine, BBox

@dataclass
class FrameLabelThresholds:
    T_ROAD: float = 0.30
    T_BIKE: float = 0.25
    T_SIDE: float = 0.30
    T_CW: float = 0.25
    shrink_frac: float = 0.08
    use_bottom_center_gate: bool = True

def _row_bbox(r) -> BBox:
    return (float(r.x1), float(r.y1), float(r.x2), float(r.y2))

def frame_space_label(bb: BBox, roi: ROIMaskEngine, thr: FrameLabelThresholds) -> str:
    bc = roi.bbox_bottom_center(bb) if thr.use_bottom_center_gate else None

    def ok(name: str, T: float) -> bool:
        if not (roi.cfg.rois.get(name) and len(roi.cfg.rois[name]) > 0):
            return False
        ov = roi.overlap_ratio(bb, name, shrink=thr.shrink_frac)
        if ov <= T:
            return False
        if thr.use_bottom_center_gate and name in {"crosswalk", "bike_lane", "sidewalk"}:
            return roi.point_in_roi(bc, name)
        return True

    # IMPORTANT priority: crosswalk must override roadway
    if ok("crosswalk", thr.T_CW):   return "crosswalk"
    if ok("bike_lane", thr.T_BIKE): return "bike_lane"
    if ok("sidewalk", thr.T_SIDE):  return "sidewalk"
    if ok("roadway", thr.T_ROAD):   return "roadway"
    return "unknown"

def label_track_frames(track_df: pd.DataFrame, roi: ROIMaskEngine, thr: FrameLabelThresholds) -> List[str]:
    bbs = track_df.apply(_row_bbox, axis=1).tolist()
    return [frame_space_label(bb, roi, thr) for bb in bbs]

def longest_run(labels: List[str], target: str) -> int:
    best = cur = 0
    for x in labels:
        if x == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

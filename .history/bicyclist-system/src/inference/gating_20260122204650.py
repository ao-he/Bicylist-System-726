# src/inference/gating.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from src.scene.roi import ROIMaskEngine, BBox

@dataclass
class GateThresholds:
    # 更宽松：bbox bottom-center 不在 ROI 时，允许一点 overlap 兜底
    overlap_fallback: float = 0.05  # 0 表示不启用
    overlap_shrink: float = 0.0

def _has_roi(roi: ROIMaskEngine, name: str) -> bool:
    return bool(roi.cfg.rois.get(name)) and len(roi.cfg.rois[name]) > 0

def _crosswalk_hit(bb: BBox, roi: ROIMaskEngine) -> bool:
    if not _has_roi(roi, "crosswalk"):
        return False
    x1, y1, x2, y2 = bb
    xs = [x1, (x1 + x2) / 2.0, x2]
    pts = [(float(x), float(y2)) for x in xs]
    return any(roi.point_in_roi(p, "crosswalk") for p in pts)

def bbox_in_study_rois(
    bb: BBox,
    roi: ROIMaskEngine,
    thr: GateThresholds = GateThresholds(),
) -> bool:
    # 0) ignore_zone 优先剔除
    bc = roi.bbox_bottom_center(bb)
    if _has_roi(roi, "ignore_zone") and roi.point_in_roi(bc, "ignore_zone"):
        return False

    # 1) crosswalk（更宽松的三点命中）
    if _crosswalk_hit(bb, roi):
        return True

    # 2) side/bike/road：bottom-center 命中即可
    for rn in ("sidewalk", "bike_lane", "roadway"):
        if _has_roi(roi, rn) and roi.point_in_roi(bc, rn):
            return True

    # 3) overlap fallback（可选，进一步放宽）
    if thr.overlap_fallback and thr.overlap_fallback > 0:
        for rn in ("crosswalk", "sidewalk", "bike_lane", "roadway"):
            if _has_roi(roi, rn):
                ov = roi.overlap_ratio(bb, rn, shrink=thr.overlap_shrink)
                if ov >= thr.overlap_fallback:
                    return True

    return False

def filter_detections_to_study_rois(
    det_df: pd.DataFrame,
    roi: ROIMaskEngine,
    thr: GateThresholds = GateThresholds(),
) -> pd.DataFrame:
    if det_df is None or len(det_df) == 0:
        return det_df
    keep = []
    for _, r in det_df.iterrows():
        bb = (float(r.x1), float(r.y1), float(r.x2), float(r.y2))
        keep.append(bbox_in_study_rois(bb, roi, thr))
    return det_df.loc[keep].copy()

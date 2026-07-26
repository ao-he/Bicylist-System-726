# src/inference/nms_lite.py
from __future__ import annotations
import pandas as pd
from typing import Tuple, List

BBox = Tuple[float, float, float, float]

def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = aa + bb - inter
    return float(inter / denom) if denom > 1e-9 else 0.0

def _row_bbox(r) -> BBox:
    return (float(r.x1), float(r.y1), float(r.x2), float(r.y2))

def nms_lite_per_frame(
    det_df: pd.DataFrame,
    frame_col: str = "frame_global",
    score_col: str = "score",
    iou_thr: float = 0.70
) -> pd.DataFrame:
    """
    A simple per-frame NMS (greedy):
      - sort by score desc
      - keep bbox if IoU with any kept bbox < iou_thr
    """
    if det_df is None or len(det_df) == 0:
        return det_df

    if score_col not in det_df.columns:
        # no score: return unchanged
        return det_df

    out = []
    for f, g in det_df.groupby(frame_col, sort=False):
        gg = g.sort_values(score_col, ascending=False)
        kept_rows = []
        kept_bbs: List[BBox] = []
        for _, r in gg.iterrows():
            bb = _row_bbox(r)
            if all(_iou(bb, kb) < iou_thr for kb in kept_bbs):
                kept_rows.append(r)
                kept_bbs.append(bb)
        if kept_rows:
            out.append(pd.DataFrame(kept_rows))

    return pd.concat(out, ignore_index=True) if out else det_df.iloc[0:0].copy()

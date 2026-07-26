from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# 你把官方 SORT 的 Sort 类放到这个路径（推荐）
# 比如：src/tracking/vendor/sort.py 里包含 class Sort
from src.tracking.vendor.sort import Sort


@dataclass
class SortParams:
    max_age: int = 3
    min_hits: int = 2
    iou_threshold: float = 0.3


class SortTracker:
    """
    Thin wrapper around the original SORT implementation.
    Converts per-frame detections to per-frame tracks.

    Input detections_df:
      columns: frame, x1, y1, x2, y2, score
      frame can be 0-based or 1-based (we treat it as labels, not index).

    Output tracks_df:
      columns: frame, track_id, x1, y1, x2, y2
      track_id is positive int (SORT returns id+1 by design).
    """

    def __init__(self, params: SortParams = SortParams()):
        self.params = params
        self.tracker = Sort(
            max_age=params.max_age,
            min_hits=params.min_hits,
            iou_threshold=params.iou_threshold,
        )

    def update_frame(self, dets_xyxy_score: np.ndarray) -> np.ndarray:
        """
        dets_xyxy_score: shape (N,5) => [x1,y1,x2,y2,score]
        returns: shape (M,5) => [x1,y1,x2,y2,track_id]
        """
        if dets_xyxy_score is None or len(dets_xyxy_score) == 0:
            dets_xyxy_score = np.empty((0, 5), dtype=float)
        return self.tracker.update(dets_xyxy_score.astype(float))

    def run(self, detections_df: pd.DataFrame) -> pd.DataFrame:
        required = {"frame", "x1", "y1", "x2", "y2", "score"}
        missing = required - set(detections_df.columns)
        if missing:
            raise ValueError(f"detections_df missing columns: {missing}")

        out_rows = []
        for frame, g in detections_df.groupby("frame", sort=True):
            dets = g[["x1", "y1", "x2", "y2", "score"]].to_numpy(dtype=float)
            trks = self.update_frame(dets)  # (M,5) [x1,y1,x2,y2,id]
            for x1, y1, x2, y2, tid in trks:
                out_rows.append(
                    {"frame": int(frame), "track_id": int(tid), "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                )

        return pd.DataFrame(out_rows)

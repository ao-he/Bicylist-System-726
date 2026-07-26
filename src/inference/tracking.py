# src/inference/tracking.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import pandas as pd

BBox = Tuple[float, float, float, float]

def _center(bb: BBox):
    x1, y1, x2, y2 = bb
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _area(bb: BBox):
    x1, y1, x2, y2 = bb
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

@dataclass
class TrackParams:
    max_dist: float = 250.0
    area_ratio_min: float = 0.20
    area_ratio_max: float = 5.00
    max_frame_gap: int = 2

def build_tracks_within_group(det_df: pd.DataFrame, params: TrackParams = TrackParams()) -> pd.DataFrame:
    """
    Build greedy tracks within a group (e.g., a super-event).
    Required columns: frame_global, x1,y1,x2,y2
    Output adds: track_id (int, local within group)
    """
    if det_df is None or len(det_df) == 0:
        return det_df.assign(track_id=pd.Series(dtype=int))

    g = det_df.sort_values("frame_global").copy()

    next_tid = 0
    active: Dict[int, Tuple[int, BBox]] = {}  # tid -> (last_frame, last_bbox)

    track_ids = []

    for _, r in g.iterrows():
        f = int(r.frame_global)
        bb = (float(r.x1), float(r.y1), float(r.x2), float(r.y2))

        # expire
        expire = [tid for tid, (lf, _) in active.items() if (f - lf) > params.max_frame_gap]
        for tid in expire:
            del active[tid]

        best_tid = None
        best_d2 = float("inf")

        a1 = _area(bb)
        c1 = _center(bb)

        for tid, (lf, last_bb) in active.items():
            a0 = _area(last_bb)
            if a0 > 1e-6 and a1 > 1e-6:
                ratio = a1 / a0
                if ratio < params.area_ratio_min or ratio > params.area_ratio_max:
                    continue
            c0 = _center(last_bb)
            d2 = (c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_tid = tid

        if best_tid is not None and best_d2 <= (params.max_dist ** 2):
            track_ids.append(best_tid)
            active[best_tid] = (f, bb)
        else:
            tid = next_tid
            next_tid += 1
            track_ids.append(tid)
            active[tid] = (f, bb)

    g["track_id"] = track_ids
    return g

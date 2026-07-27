# src/inference/rider_counting_new.py
"""
Per-rider counting for burst-capture data (method B: detect every frame,
then de-duplicate).

Counting logic:
  - Every detection is a rider *appearance*.
  - Appearances in nearby captures (image-number gap <= max_frame_gap) that
    are spatially consistent are associated into one rider.
  - Appearances in distant captures are always different riders: at ~2 s per
    capture pair a cyclist leaves the scene within a few captures, so no
    long-range association is attempted.  This is what makes the method safe
    on sparse scenes (association degenerates to per-frame counting) while
    still de-duplicating the common same-rider burst pairs.

Direction:
  - Riders with >= 2 associated appearances get a provisional direction from
    the displacement of the bbox bottom center (first -> last appearance).
  - Single-appearance riders are left direction-unknown; the `orientation`
    column is reserved for a single-frame appearance-based direction label
    (manual review or a vision-model classifier) filled in downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

BBox = Tuple[float, float, float, float]


@dataclass
class AssocParams:
    max_frame_gap: int = 3        # image-number gap allowed within one rider
    max_dist_factor: float = 3.0  # gate = factor * mean bbox diagonal
    min_gate_px: float = 80.0     # lower bound for the distance gate
    min_move_px: float = 8.0      # displacement needed for a direction call
    cos_gate: float = 0.5         # |cos| below this = crossing, not along/against


def _bottom_center(bb: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bb
    return ((x1 + x2) / 2.0, y2)


def _diag(bb: BBox) -> float:
    x1, y1, x2, y2 = bb
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def associate_riders(det_df: pd.DataFrame, params: AssocParams = AssocParams()) -> pd.DataFrame:
    """
    Assign a rider_id to every detection.

    Required columns: img_num, x1, y1, x2, y2.
    Detections within the same image are never merged (two boxes in one
    frame are two riders by definition — same-frame duplicates should be
    removed by NMS before this step).
    """
    if det_df is None or len(det_df) == 0:
        return det_df.assign(rider_id=pd.Series(dtype=int))

    g = det_df.sort_values(["img_num"]).copy()
    next_rid = 0
    # rid -> (last_img_num, last_bbox)
    active: Dict[int, Tuple[int, BBox]] = {}
    rider_ids: List[int] = []

    for img_num, frame in g.groupby("img_num", sort=True):
        # expire riders that are too far in the past
        for rid in [r for r, (n, _) in active.items()
                    if img_num - n > params.max_frame_gap]:
            del active[rid]

        rows = list(frame.itertuples())
        bbs = [(float(r.x1), float(r.y1), float(r.x2), float(r.y2)) for r in rows]

        # greedy one-to-one matching: closest pairs first
        cands = []
        for i, bb in enumerate(bbs):
            c = _bottom_center(bb)
            gate = max(params.min_gate_px, params.max_dist_factor * _diag(bb))
            for rid, (_, last_bb) in active.items():
                lc = _bottom_center(last_bb)
                d = ((c[0] - lc[0]) ** 2 + (c[1] - lc[1]) ** 2) ** 0.5
                if d <= gate:
                    cands.append((d, i, rid))
        cands.sort()
        assigned_det: Dict[int, int] = {}
        used_rids = set()
        for d, i, rid in cands:
            if i in assigned_det or rid in used_rids:
                continue
            assigned_det[i] = rid
            used_rids.add(rid)

        for i, bb in enumerate(bbs):
            rid = assigned_det.get(i)
            if rid is None:
                rid = next_rid
                next_rid += 1
            rider_ids.append(rid)
            active[rid] = (int(img_num), bb)

    g["rider_id"] = rider_ids
    return g


def summarize_riders(
    det_df: pd.DataFrame,
    flow: Optional[Tuple[float, float]],
    params: AssocParams = AssocParams(),
) -> pd.DataFrame:
    """
    One row per rider. Requires columns: rider_id, img, img_num, x1..y2,
    space_label. Facility flags use the same any-involvement semantics as
    the event pipeline (crosswalk counts as roadway).
    """
    rows = []
    for rid, g in det_df.groupby("rider_id", sort=True):
        g = g.sort_values("img_num")
        labels = g["space_label"].tolist()

        first, last = g.iloc[0], g.iloc[-1]
        bc0 = _bottom_center((first.x1, first.y1, first.x2, first.y2))
        bc1 = _bottom_center((last.x1, last.y1, last.x2, last.y2))
        dx, dy = bc1[0] - bc0[0], bc1[1] - bc0[1]
        disp = (dx * dx + dy * dy) ** 0.5

        direction = "unknown"
        cos = None
        wrong_way: Optional[bool] = None
        if flow is not None and len(g) >= 2 and disp >= params.min_move_px:
            ux, uy = dx / disp, dy / disp
            cos = ux * float(flow[0]) + uy * float(flow[1])
            if abs(cos) < params.cos_gate:
                # movement is mostly perpendicular to the reference flow:
                # a crossing maneuver, for which wrong-way is undefined
                direction = "cross_flow"
            else:
                direction = "along_flow" if cos >= 0 else "against_flow"
                wrong_way = cos < 0

        rows.append({
            "rider_id": rid,
            "n_obs": len(g),
            "img_first": first.img,
            "img_last": last.img,
            "img_num_first": int(first.img_num),
            "img_num_last": int(last.img_num),
            "in_sidewalk_any": "sidewalk" in labels,
            "in_bike_lane_any": "bike_lane" in labels,
            "in_roadway_any": ("roadway" in labels) or ("crosswalk" in labels),
            "disp_px": round(disp, 1),
            "direction_displacement": direction,
            "cos_to_flow": None if cos is None else round(float(cos), 3),
            "wrong_way_displacement": wrong_way,
            "orientation": None,   # filled downstream (manual / vision model)
        })
    return pd.DataFrame(rows)

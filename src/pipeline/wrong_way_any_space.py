# src/inference/wrong_way_any_space.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd

from src.scene.roi import ROIMaskEngine
from src.inference.common import FrameLabelThresholds, direction_cosine


@dataclass
class WrongWayAnySpaceThresholds(FrameLabelThresholds):
    min_track_frames: int = 3
    min_move_px: float = 8.0


def infer_wrong_way_any_space_per_track(
    tracks_df: pd.DataFrame,
    roi: ROIMaskEngine,
    thr: WrongWayAnySpaceThresholds = WrongWayAnySpaceThresholds(),
) -> Tuple[pd.DataFrame, Dict]:
    """
    Wrong-way (any-space):
      - requires flow_vector exists
      - requires track has enough motion (min_move_px)
      - wrong_way = direction == against_flow
    """
    required = {"track_id", "x1", "y1", "x2", "y2"}
    missing = required - set(tracks_df.columns)
    if missing:
        raise ValueError(f"tracks_df missing columns: {missing}")

    flow = roi.flow_vector()
    scene_applicable = (flow is not None)

    rows = []
    for tid, g in tracks_df.groupby("track_id", sort=False):
        if (not scene_applicable) or (len(g) < thr.min_track_frames):
            rows.append({"track_id": tid, "direction": pd.NA, "cos_to_flow": pd.NA, "wrong_way": pd.NA})
            continue

        direction, cos = direction_cosine(g, roi, min_move_px=thr.min_move_px)
        if direction == "unknown" or cos is None:
            rows.append({"track_id": tid, "direction": pd.NA, "cos_to_flow": pd.NA, "wrong_way": pd.NA})
            continue

        wrong_way = (direction == "against_flow")
        rows.append({"track_id": tid, "direction": direction, "cos_to_flow": float(cos), "wrong_way": bool(wrong_way)})

    df = pd.DataFrame(rows)
    app = df["wrong_way"].notna()
    denom = int(app.sum())
    num = int((df.loc[app, "wrong_way"] == True).sum()) if denom > 0 else 0

    summary = {
        "N_tracks": int(len(df)),
        "N_applicable": denom,
        "N_wrong_way": num,
        "share_wrong_way": (num / denom) if denom > 0 else None,
        "scene_applicable": bool(scene_applicable),
    }
    return df, summary

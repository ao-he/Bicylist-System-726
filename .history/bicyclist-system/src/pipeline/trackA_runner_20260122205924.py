
# src/pipelines/trackA_runner.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Union
import pandas as pd

from src.scene.roi import load_roi_config, ROIMaskEngine

from src.inference.gating import filter_detections_to_study_roi, GateThresholds
from src.inference.nms_lite import nms_lite_per_frame
from src.inference.tracking import build_tracks_within_group, TrackParams

from src.inference.crossing import infer_crossing_per_track, CrossingThresholds
from src.inference.space_usage import infer_space_usage_per_track, SpaceUsageThresholds
from src.inference.wrong_way_any_space import infer_wrong_way_any_space_per_track, WrongWayAnySpaceThresholds


def _normalize_super_events(super_events: List[List[Union[str, Path]]]) -> List[List[str]]:
    out = []
    for ev in super_events:
        names = []
        for x in ev:
            if isinstance(x, Path):
                names.append(x.name)
            else:
                # could be full path or filename
                names.append(Path(str(x)).name)
        out.append(names)
    return out


def run_trackA_from_detections_and_super_events(
    location_id: str,
    roi_json_path: Union[str, Path],
    detections_df: pd.DataFrame,
    super_events: List[List[Union[str, Path]]],
    outdir: Union[str, Path],
    *,
    # knobs (safe defaults)
    gate_overlap_fallback: float = 0.05,
    nms_iou_thr: float = 0.70,
    track_max_dist: float = 250.0,
    track_max_gap: int = 2,
    min_track_frames: int = 5,
    min_consecutive_frames: int = 3,
    min_move_px: float = 8.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Inputs:
      detections_df columns MUST include:
        img (filename), frame_global, x1,y1,x2,y2
      Optional:
        score (for NMS-lite)

      super_events: list of list of image filenames (or Paths)

    Outputs:
      outdir/tracks.csv
      outdir/per_track_metrics.csv
      outdir/trackA_summary.json
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    roi_cfg = load_roi_config(Path(roi_json_path))
    roi = ROIMaskEngine(roi_cfg)

    # ---------- 0) Basic column check ----------
    required = {"img", "frame_global", "x1", "y1", "x2", "y2"}
    missing = required - set(detections_df.columns)
    if missing:
        raise ValueError(f"detections_df missing columns: {missing}")

    # ---------- 1) optional same-frame de-dup (fix duplicate boxes) ----------
    det = nms_lite_per_frame(detections_df, frame_col="frame_global", score_col="score", iou_thr=nms_iou_thr)

    # ---------- 2) ROI gate: only keep detections in study ROI union ----------
    det = filter_detections_to_study_roi(det, roi, GateThresholds(overlap_fallback=gate_overlap_fallback))

    # ---------- 3) build tracks within each super-event ----------
    super_events_norm = _normalize_super_events(super_events)

    all_tracks = []
    for sid, img_names in enumerate(super_events_norm):
        d = det[det["img"].isin(img_names)].copy()
        if len(d) == 0:
            continue

        d = build_tracks_within_group(d, TrackParams(
            max_dist=track_max_dist,
            max_frame_gap=track_max_gap
        ))
        d["super_event_id"] = sid

        # make track_id global-unique (sid * big_stride + local_tid)
        d["track_id"] = d["track_id"].astype(int) + sid * 10000
        all_tracks.append(d)

    tracks_df = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame(
        columns=list(det.columns) + ["track_id", "super_event_id"]
    )

    tracks_df.to_csv(outdir / "tracks.csv", index=False)

    # ---------- 4) infer 3 behaviors (per track) ----------
    crossing_thr = CrossingThresholds(min_track_frames=min_track_frames, min_consecutive_frames=min_consecutive_frames)
    space_thr = SpaceUsageThresholds(min_track_frames=min_track_frames)
    ww_thr = WrongWayAnySpaceThresholds(min_track_frames=min_track_frames, min_move_px=min_move_px)

    crossing_df, crossing_sum = infer_crossing_per_track(tracks_df, roi, crossing_thr)
    space_df, space_sum = infer_space_usage_per_track(tracks_df, roi, space_thr)
    ww_df, ww_sum = infer_wrong_way_any_space_per_track(tracks_df, roi, ww_thr)

    per_track = (crossing_df
                 .merge(space_df, on="track_id", how="outer")
                 .merge(ww_df, on="track_id", how="outer"))

    per_track.to_csv(outdir / "per_track_metrics.csv", index=False, na_rep="NA")

    summary = {
        "location_id": location_id,
        "N_tracks": int(per_track.shape[0]),
        "crossing": crossing_sum,
        "space_usage": space_sum,
        "wrong_way_any_space": ww_sum,
        "params": {
            "gate_overlap_fallback": gate_overlap_fallback,
            "nms_iou_thr": nms_iou_thr,
            "track_max_dist": track_max_dist,
            "track_max_gap": track_max_gap,
            "min_track_frames": min_track_frames,
            "min_consecutive_frames": min_consecutive_frames,
            "min_move_px": min_move_px,
        }
    }
    (outdir / "trackA_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return per_track, summary

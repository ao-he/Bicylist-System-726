# scripts/run_location_from_detections.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.scene.roi import load_roi_config, ROIMaskEngine
from src.tracking.sort_tracker import SortTracker, SortParams

from src.inference.crossing import infer_crossing_per_track
from src.inference.wrong_way import infer_wrong_way_per_track
from src.inference.right_side import infer_right_side_per_track
from src.inference.space_usage import infer_space_usage_per_track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True, help="location id (used for folder name), e.g., loc_01")
    ap.add_argument("--roi", required=False, default=None,
                    help="ROI JSON path. default: configs/locations/<loc>.json")
    ap.add_argument("--detections", required=True,
                    help="detections csv: frame,x1,y1,x2,y2,score")
    ap.add_argument("--outdir", default="outputs", help="output folder")
    ap.add_argument("--max_age", type=int, default=3)
    ap.add_argument("--min_hits", type=int, default=1)
    ap.add_argument("--iou_thr", type=float, default=0.3)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    roi_path = Path(args.roi) if args.roi else (repo_root / "configs" / "locations" / f"{args.loc}.json")
    det_path = Path(args.detections)

    if not roi_path.exists():
        raise FileNotFoundError(f"ROI JSON not found: {roi_path}")
    if not det_path.exists():
        raise FileNotFoundError(f"detections csv not found: {det_path}")

    # load ROI
    cfg = load_roi_config(roi_path)
    roi = ROIMaskEngine(cfg)

    # load detections
    det_df = pd.read_csv(det_path)
    required = {"frame", "x1", "y1", "x2", "y2", "score"}
    missing = required - set(det_df.columns)
    if missing:
        raise ValueError(f"detections csv missing columns: {missing}")

    # run SORT
    tracker = SortTracker(SortParams(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_thr))
    tracks_df = tracker.run(det_df)

    # run inference modules
    cross_df, cross_sum = infer_crossing_per_track(tracks_df, roi)
    ww_df, ww_sum       = infer_wrong_way_per_track(tracks_df, roi)
    rs_df, rs_sum       = infer_right_side_per_track(tracks_df, roi)
    sp_df, sp_sum       = infer_space_usage_per_track(tracks_df, roi)

    # merge cleanly (avoid _x/_y explosion)
    events_all = (
        cross_df
        .merge(ww_df[["track_id", "in_bike_lane_any", "direction", "cos_to_flow", "wrong_way"]],
               on="track_id", how="left")
        .merge(rs_df[["track_id", "in_roadway_any", "in_right_band_any", "direction", "cos_to_flow", "right_side_violation"]],
               on="track_id", how="left", suffixes=("", "_rs"))
        .merge(sp_df[["track_id", "dominant_space"]],
               on="track_id", how="left")
    )

    # combined summary
    summary_all = {
        "loc": args.loc,
        "roi_path": str(roi_path),
        "detections_path": str(det_path),
        "sort": {"max_age": args.max_age, "min_hits": args.min_hits, "iou_threshold": args.iou_thr},
        "crossing": cross_sum,
        "wrong_way": ww_sum,
        "right_side": rs_sum,
        "space_usage": sp_sum,
    }

    outdir = Path(args.outdir) / args.loc
    outdir.mkdir(parents=True, exist_ok=True)

    tracks_df.to_csv(outdir / "tracks.csv", index=False)
    events_all.to_csv(outdir / "events_all.csv", index=False)
    (outdir / "summary_all.json").write_text(json.dumps(summary_all, indent=2), encoding="utf-8")

    print(f"[OK] wrote {outdir/'tracks.csv'}")
    print(f"[OK] wrote {outdir/'events_all.csv'}")
    print(f"[OK] wrote {outdir/'summary_all.json'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Union, Optional
from pathlib import Path
import pandas as pd

from src.scene.roi import load_roi_config, ROIMaskEngine
from src.inference.common import frame_space_label, FrameLabelThresholds


@dataclass
class EventAParams:
    # labeling
    bottom_edge_npts: int = 5
    vote_min_pts: int = 2

    # wrong-way
    min_move_px: float = 8.0

    # which ROIs are considered "study area"
    keep_labels: tuple = ("sidewalk", "bike_lane", "roadway", "crosswalk")


def bbox_bottom_center(bb):
    x1, y1, x2, y2 = bb
    return ((x1 + x2) / 2.0, y2)


def run_eventA_from_detections_and_events(
    location_id: str,
    roi_json_path: Union[str, Path],
    img_paths: List[Path],
    detections_df: pd.DataFrame,
    events: List[List[Union[str, Path]]],
    outdir: Union[str, Path],
    params: EventAParams = EventAParams(),
) -> tuple[pd.DataFrame, Dict]:
    """
    Frame -> Event aggregation pipeline.
    - wide detection (input)
    - frame-level ROI label (hard assignment by bottom-edge voting)
    - event-level occupancy + wrong-way
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_roi_config(roi_json_path)
    roi = ROIMaskEngine(cfg)
    flow = roi.flow_vector()

    img2idx = {p.name: i for i, p in enumerate(img_paths)}

    # --- frame-level labeling ---
    thr = FrameLabelThresholds(bottom_edge_npts=params.bottom_edge_npts, vote_min_pts=params.vote_min_pts)

    d = detections_df.copy()
    if len(d) == 0:
        events_df = pd.DataFrame([{"event_id": i, "has_target": False} for i in range(len(events))])
        summary = {"location_id": location_id, "N_events": len(events), "N_has_target": 0}
        return events_df, summary

    # ensure frame_global exists
    if "frame_global" not in d.columns:
        d["frame_global"] = d["img"].map(img2idx)

    def _label_row(r):
        bb = (float(r.x1), float(r.y1), float(r.x2), float(r.y2))
        return frame_space_label(bb, roi, thr)

    d["space_label"] = d.apply(_label_row, axis=1)
    d = d[d["space_label"].isin(params.keep_labels)].copy()

    # save labeled detections (optional, but very useful)
    d.to_csv(outdir / "detections_labeled.csv", index=False)

    # --- event-level aggregation ---
    rows = []
    for eid, ev in enumerate(events):
        ev_names = [p.name if isinstance(p, Path) else str(p) for p in ev]
        det_ev = d[d["img"].isin(ev_names)].copy()

        if len(det_ev) == 0:
            rows.append({"event_id": eid, "n_imgs": len(ev), "has_target": False})
            continue

        # occupancy counts
        c_side = int((det_ev["space_label"] == "sidewalk").sum())
        c_bike = int((det_ev["space_label"] == "bike_lane").sum())
        c_road = int((det_ev["space_label"] == "roadway").sum())
        c_cw   = int((det_ev["space_label"] == "crosswalk").sum())
        total  = c_side + c_bike + c_road + c_cw

        in_side = c_side > 0
        in_bike = c_bike > 0
        in_road = (c_road + c_cw) > 0

        # dominant (majority vote)
        dom = "unknown"
        if total > 0:
            dom = max(
                [("sidewalk", c_side), ("bike_lane", c_bike), ("roadway", c_road + c_cw)],
                key=lambda x: x[1]
            )[0]

        # wrong-way (earliest & latest valid frames inside study ROIs)
        wrong_way = pd.NA
        direction = pd.NA
        cos_to_flow = pd.NA

        if (flow is not None) and (len(det_ev) >= 2):
            fg0 = int(det_ev["frame_global"].min())
            fg1 = int(det_ev["frame_global"].max())

            det0 = det_ev[det_ev["frame_global"] == fg0].sort_values("score", ascending=False).head(1)
            det1 = det_ev[det_ev["frame_global"] == fg1].sort_values("score", ascending=False).head(1)

            if len(det0) == 1 and len(det1) == 1:
                bb0 = (float(det0.iloc[0].x1), float(det0.iloc[0].y1), float(det0.iloc[0].x2), float(det0.iloc[0].y2))
                bb1 = (float(det1.iloc[0].x1), float(det1.iloc[0].y1), float(det1.iloc[0].x2), float(det1.iloc[0].y2))
                bc0 = bbox_bottom_center(bb0)
                bc1 = bbox_bottom_center(bb1)

                vx, vy = bc1[0] - bc0[0], bc1[1] - bc0[1]
                mag = float((vx * vx + vy * vy) ** 0.5)

                if mag >= params.min_move_px:
                    ux, uy = vx / mag, vy / mag
                    cos = float(ux * float(flow[0]) + uy * float(flow[1]))
                    cos_to_flow = cos
                    direction = "along_flow" if cos >= 0 else "against_flow"
                    wrong_way = bool(cos < 0)

        rows.append({
            "event_id": eid,
            "n_imgs": len(ev),
            "has_target": True,
            "n_det": int(total),

            "in_sidewalk_any": bool(in_side),
            "in_bike_lane_any": bool(in_bike),
            "in_roadway_any": bool(in_road),
            "dominant_space": dom,

            "wrong_way": wrong_way,
            "direction": direction,
            "cos_to_flow": cos_to_flow,
        })

    events_df = pd.DataFrame(rows)
    events_df.to_csv(outdir / "events_behavior.csv", index=False)

    # --- summary (applicable only for wrong-way) ---
    has = events_df["has_target"] == True
    applicable = events_df["wrong_way"].notna()
    N_app = int(applicable.sum())
    N_ww = int((events_df.loc[applicable, "wrong_way"] == True).sum()) if N_app > 0 else 0

    summary = {
        "location_id": location_id,
        "N_events": int(len(events_df)),
        "N_has_target": int(has.sum()),
        "N_wrong_way_applicable": N_app,
        "N_wrong_way": N_ww,
        "share_wrong_way": (N_ww / N_app) if N_app > 0 else None,
        "params": {
            "bottom_edge_npts": params.bottom_edge_npts,
            "vote_min_pts": params.vote_min_pts,
            "min_move_px": params.min_move_px,
        }
    }
    (outdir / "summary.json").write_text(pd.io.json.dumps(summary, indent=2), encoding="utf-8")

    return events_df, summary

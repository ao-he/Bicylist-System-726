#!/usr/bin/env python3
"""
Rider counting pipeline (method B): detect every capture, de-duplicate
nearby captures into riders, label facilities with the (re-annotated) ROI
configs, and report per-scene counts.

Run locally next to the raw data, one location at a time:

    python scripts/run_rider_count_new.py \
        --loc-id loc_25 \
        --img-dir "C:/Users/78222/Desktop/28_locations/0_MAIN_BIKE_DATASETS_clean/Loc_25/Bicyclist" \
        --outdir outputs_new/loc_25

or over every location that has a ROI config:

    python scripts/run_rider_count_new.py --batch \
        --data-root "C:/Users/78222/Desktop/28_locations/0_MAIN_BIKE_DATASETS_clean"

Outputs per location (in --outdir):
    detections_raw.csv   every detection (explicit conf threshold)
    detections_riders.csv detections + space_label + rider_id
    riders.csv           one row per rider (counting unit)
    scene_summary.json   counts + parameters
    integrity_report.csv unreadable / truncated captures
    crops/               (--save-crops) one crop per appearance, for
                         orientation labeling
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.scene.roi_new import load_roi_config, ExclusiveROIMaskEngine
from src.inference.common import frame_space_label, FrameLabelThresholds
from src.inference.nms_lite import nms_lite_per_frame

KEEP_LABELS = {"sidewalk", "bike_lane", "roadway", "crosswalk"}

# ---------------------------------------------------------------------------
# Rider association & summary (self-contained: no external _new module needed)
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

BBox = Tuple[float, float, float, float]



@dataclass
class AssocParams:
    max_frame_gap: int = 3        # image-number gap allowed within one rider
    max_dist_factor: float = 3.0  # gate = factor * mean bbox diagonal
    min_gate_px: float = 80.0     # lower bound for the distance gate
    burst_rescue: bool = True     # burst pairs (gap <= rescue_gap) with a single
    rescue_gap: int = 2           # detection on each side associate regardless of
                                  # distance: a fast rider crosses most of the frame
                                  # between 1s captures
    rescue_max_frac: float = 0.8  # but never farther than this fraction of the
                                  # frame width (guards against exit/enter merges)
    min_move_px: float = 25.0     # calibrated: stationary-target jitter is <13px, real riders move >50px between captures
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
    frame_width = float(g["x2"].max())
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

        # burst rescue: single detection, single recent rider -> same person,
        # even when the fast-rider displacement exceeds the spatial gate
        if params.burst_rescue and len(bbs) == 1 and not assigned_det:
            recent = [rid for rid, (n, _) in active.items()
                      if img_num - n <= params.rescue_gap and rid not in used_rids]
            if len(recent) == 1:
                c = _bottom_center(bbs[0])
                lc = _bottom_center(active[recent[0]][1])
                if abs(c[0] - lc[0]) <= params.rescue_max_frac * frame_width:
                    assigned_det[0] = recent[0]

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




def dedup_contained_per_frame(det, frame_col="img_num", iomin_thr=0.60):
    """Suppress same-frame boxes largely contained in a higher-score box.
    Catches duplicate detections of one bicycle (partial box inside full box)
    that plain IoU-NMS misses."""
    if det is None or len(det) == 0:
        return det
    keep_idx = []
    for _, g in det.groupby(frame_col, sort=False):
        g = g.sort_values("score", ascending=False)
        kept = []
        for idx, r in g.iterrows():
            a = (r.x1, r.y1, r.x2, r.y2)
            aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
            dup = False
            for kr in kept:
                b = (kr.x1, kr.y1, kr.x2, kr.y2)
                iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = iw * ih
                bb = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
                if inter / max(min(aa, bb), 1e-6) >= iomin_thr:
                    dup = True
                    break
            if not dup:
                kept.append(r)
                keep_idx.append(idx)
    return det.loc[keep_idx]


def imgnum(name: str) -> int:
    m = re.search(r"(\d+)", Path(str(name)).stem)
    return int(m.group(1)) if m else -1


def collect_images(img_dir: Path, max_images=None):
    paths = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda p: imgnum(p.name),
    )
    return paths[:max_images] if max_images else paths


def read_image(path: Path):
    """cv2 first; fall back to PIL with truncated-JPEG tolerance."""
    import cv2
    img = cv2.imread(str(path))
    if img is not None:
        return img, "ok"
    try:
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(path) as im:
            arr = np.array(im.convert("RGB"))[:, :, ::-1].copy()
        return arr, "truncated_recovered"
    except Exception as e:
        return None, f"unreadable: {e}"


def run_location(loc_id, img_dir, roi_json, outdir, args):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_roi_config(roi_json)
    roi = ExclusiveROIMaskEngine(cfg)
    flow = roi.flow_vector()

    img_paths = collect_images(Path(img_dir), getattr(args, "max_images", None))
    if not img_paths:
        print(f"[{loc_id}] no images in {img_dir}")
        return None
    print(f"[{loc_id}] {len(img_paths)} captures, flow={'yes' if flow else 'none'}")

    from ultralytics import YOLO
    model = YOLO(args.model)

    det_rows, person_rows, integrity = [], [], []
    for i, p in enumerate(img_paths):
        img, status = read_image(p)
        if img is None:
            integrity.append({"img": p.name, "status": status})
            continue
        if status != "ok":
            integrity.append({"img": p.name, "status": status})
        r = model(img, conf=args.conf, imgsz=getattr(args, "imgsz", 640), verbose=False)[0]
        if r.boxes is None:
            continue
        for (x1, y1, x2, y2), s, c in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
        ):
            row = {
                "img": p.name, "img_num": imgnum(p.name), "frame_global": i,
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "score": float(s), "cls": int(c),
            }
            if c in args.classes:
                det_rows.append(row)
            elif c == 0 and getattr(args, "person_probe", True):
                # person without a counted class: candidate for a missed rider
                person_rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"[{loc_id}]   {i+1}/{len(img_paths)} captures, {len(det_rows)} detections")

    det = pd.DataFrame(det_rows)
    det.to_csv(outdir / "detections_raw.csv", index=False)
    n_det_raw = len(det)
    pd.DataFrame(integrity).to_csv(outdir / "integrity_report.csv", index=False)

    if len(det) == 0:
        summary = {"location_id": loc_id, "n_captures": len(img_paths), "n_riders": 0}
        (outdir / "scene_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{loc_id}] no detections")
        return summary

    det = nms_lite_per_frame(det, frame_col="img_num", score_col="score", iou_thr=getattr(args, "nms_iou", 0.70))
    n_after_nms = len(det)
    det = dedup_contained_per_frame(det, iomin_thr=getattr(args, "iomin_thr", 0.60))
    n_after_dedup = len(det)

    thr = FrameLabelThresholds(bottom_edge_npts=5, vote_min_pts=2)
    det["space_label"] = det.apply(
        lambda r: frame_space_label((r.x1, r.y1, r.x2, r.y2), roi, thr), axis=1)
    det = det[det["space_label"].isin(KEEP_LABELS)].copy()
    n_in_roi = len(det)

    assoc = AssocParams(max_frame_gap=getattr(args, "assoc_gap", 3),
                        min_move_px=getattr(args, "min_move_px", 25.0),
                        cos_gate=getattr(args, "cos_gate", 0.5))
    det = associate_riders(det, assoc)
    det.to_csv(outdir / "detections_riders.csv", index=False)

    riders = summarize_riders(det, flow, assoc)
    riders.to_csv(outdir / "riders.csv", index=False)

    person_df = pd.DataFrame(person_rows)
    if len(person_df):
        bike_imgs = set(det["img"]) if len(det) else set()
        person_df["frame_has_bicycle"] = person_df["img"].isin(bike_imgs)
        person_df["space_label"] = person_df.apply(
            lambda r: frame_space_label((r.x1, r.y1, r.x2, r.y2), roi, thr), axis=1)
        person_df.to_csv(outdir / "person_candidates.csv", index=False)
        n_missed = int((~person_df["frame_has_bicycle"]).groupby(person_df["img"]).any().sum())
        print(f"[{loc_id}] person candidates: {len(person_df)} dets, "
              f"{n_missed} frames with person but NO bicycle (possible missed riders)")
        if getattr(args, "save_crops", False):
            import cv2
            pc_dir = outdir / "crops_person"
            pc_dir.mkdir(exist_ok=True)
            by_img = {p.name: p for p in img_paths}
            for _, r in person_df[~person_df["frame_has_bicycle"]].iterrows():
                img, _ = read_image(by_img[r.img])
                if img is None:
                    continue
                h, w = img.shape[:2]
                px = int(0.2 * (r.x2 - r.x1)); py = int(0.2 * (r.y2 - r.y1))
                x1 = max(0, int(r.x1) - px); y1 = max(0, int(r.y1) - py)
                x2 = min(w, int(r.x2) + px); y2 = min(h, int(r.y2) + py)
                cv2.imwrite(str(pc_dir / f"{Path(r.img).stem}_p{int(r.x1)}.jpg"), img[y1:y2, x1:x2])

    if getattr(args, "save_viz", True) and (len(det) or len(person_df)):
        img_lookup = {p.name: p for p in img_paths}
        n_viz = render_visualizations(det, riders, cfg, img_lookup, outdir / "viz",
                                      person_det=person_df)
        print(f"[{loc_id}] viz: {n_viz} annotated captures -> {outdir/'viz'}")

    if getattr(args, "save_viz", True) and len(det):
        n_dc = render_direction_checks(det, riders, cfg,
                                       {p.name: p for p in img_paths},
                                       outdir / "direction_check")
        if n_dc:
            print(f"[{loc_id}] direction checks: {n_dc} verification images -> {outdir/'direction_check'}")

    if getattr(args, "save_crops", False) and len(det):
        import cv2
        crop_dir = outdir / "crops"
        crop_dir.mkdir(exist_ok=True)
        by_img = {p.name: p for p in img_paths}
        for _, r in det.iterrows():
            img, _ = read_image(by_img[r.img])
            if img is None:
                continue
            h, w = img.shape[:2]
            pad_x = int(0.15 * (r.x2 - r.x1)); pad_y = int(0.15 * (r.y2 - r.y1))
            x1 = max(0, int(r.x1) - pad_x); y1 = max(0, int(r.y1) - pad_y)
            x2 = min(w, int(r.x2) + pad_x); y2 = min(h, int(r.y2) + pad_y)
            cv2.imwrite(str(crop_dir / f"rider{int(r.rider_id):04d}_{Path(r.img).stem}.jpg"),
                        img[y1:y2, x1:x2])

    n_along = int((riders["direction_displacement"] == "along_flow").sum())
    n_against = int((riders["direction_displacement"] == "against_flow").sum())
    n_cross = int((riders["direction_displacement"] == "cross_flow").sum())
    n_dir = n_along + n_against
    n_ww = int((riders["wrong_way_displacement"] == True).sum())
    summary = {
        "location_id": loc_id,
        "n_captures": int(len(img_paths)),
        "n_unreadable": int(sum(1 for r in integrity if str(r["status"]).startswith("unreadable"))),
        "n_detections": int(len(det)),
        "n_riders": int(riders.shape[0]),
        "n_person_candidates": int(len(person_rows)),
        "n_riders_multi_obs": int((riders["n_obs"] >= 2).sum()),
        "riders_in_sidewalk": int(riders["in_sidewalk_any"].sum()),
        "riders_in_bike_lane": int(riders["in_bike_lane_any"].sum()),
        "riders_in_roadway": int(riders["in_roadway_any"].sum()),
        "direction_known_displacement": n_dir,
        "riders_crossing_flow": n_cross,
        "wrong_way_displacement": n_ww,
        "note": "wrong_way is provisional (displacement only); single-obs riders await orientation labels",
        "funnel": {
            "captures": int(len(img_paths)),
            "unreadable": int(sum(1 for r in integrity if str(r["status"]).startswith("unreadable"))),
            "truncated_recovered": int(sum(1 for r in integrity if r["status"] == "truncated_recovered")),
            "detections_raw": int(n_det_raw),
            "after_nms": int(n_after_nms),
            "after_containment_dedup": int(n_after_dedup),
            "in_roi": int(n_in_roi),
            "riders": int(riders.shape[0]),
            "person_candidates": int(len(person_rows)),
        },
        "params": {
            "model": args.model, "conf": args.conf, "classes": sorted(args.classes),
            "nms_iou": getattr(args, "nms_iou", 0.70), "assoc_max_frame_gap": getattr(args, "assoc_gap", 3),
            "min_move_px": getattr(args, "min_move_px", 25.0), "cos_gate": getattr(args, "cos_gate", 0.5), "roi_exclusive": True,
        },
    }
    (outdir / "scene_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{loc_id}] riders={summary['n_riders']} "
          f"(sidewalk {summary['riders_in_sidewalk']}, bike {summary['riders_in_bike_lane']}, "
          f"road {summary['riders_in_roadway']}), dir-known {n_dir}, cross {n_cross}, ww {n_ww}")
    return summary



VIZ_COLORS = {  # BGR
    "roadway": (60, 60, 230), "bike_lane": (80, 190, 60),
    "sidewalk": (220, 190, 40), "crosswalk": (230, 60, 230),
}



def render_direction_checks(det, riders, cfg, img_lookup, out_dir):
    """One side-by-side (first | last capture) image per direction-judged rider,
    with bboxes, the displacement arrow, the reference flow arrow and the
    verdict banner — for manual verification of every wrong-way call."""
    import cv2
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    judged = riders[riders["direction_displacement"].isin(
        ["along_flow", "against_flow", "cross_flow"])]
    for _, R in judged.iterrows():
        g = det[det["rider_id"] == R.rider_id].sort_values("img_num")
        first, last = g.iloc[0], g.iloc[-1]
        panels = []
        for r in (first, last):
            src = img_lookup.get(r.img)
            img, _ = (read_image(src) if src else (None, None))
            if img is None:
                break
            img = img.copy()
            cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)),
                          (0, 165, 255), 4)
            panels.append((img, r))
        if len(panels) != 2:
            continue
        h = min(p_[0].shape[0] for p_ in panels)
        scaled = []
        for img, r in panels:
            sc = h / img.shape[0]
            scaled.append((cv2.resize(img, (int(img.shape[1] * sc), h)), r, sc))
        comp = cv2.hconcat([s_[0] for s_ in scaled])
        # displacement arrow in TRUE scene coordinates, drawn inside the left panel
        (i0, r0, s0), (i1, r1, s1) = scaled
        p0 = (int((r0.x1 + r0.x2) / 2 * s0), int(r0.y2 * s0))
        p1 = (int((r1.x1 + r1.x2) / 2 * s0), int(r1.y2 * s0))
        cv2.arrowedLine(comp, p0, p1, (255, 0, 255), 6, tipLength=0.08)
        # ghost circle marking where the rider ends up (also on the right panel)
        cv2.circle(comp, p1, 14, (255, 0, 255), 3)
        pr = (int((r1.x1 + r1.x2) / 2 * s1) + i0.shape[1], int(r1.y2 * s1))
        cv2.circle(comp, pr, 14, (255, 0, 255), 3)
        # flow arrow on left panel
        if cfg.flow and cfg.flow.get("vector"):
            v = cfg.flow["vector"]
            tip = (int(v["x2"] * s0), int(v["y2"] * s0))
            cv2.arrowedLine(comp, (int(v["x1"] * s0), int(v["y1"] * s0)), tip,
                            (0, 165, 255), 4, tipLength=0.06)
            for th, col in ((6, (0, 0, 0)), (2, (0, 165, 255))):
                cv2.putText(comp, "FLOW", (tip[0] - 40, tip[1] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, th, cv2.LINE_AA)
        # verdict banner
        ww = R.wrong_way_displacement
        verdict = ("WRONG-WAY" if ww is True else
                   ("crossing" if R.direction_displacement == "cross_flow" else "with flow"))
        color = (60, 60, 230) if ww is True else ((230, 160, 40) if verdict == "crossing" else (60, 180, 60))
        banner = comp.copy()
        cv2.rectangle(banner, (0, 0), (comp.shape[1], 64), (250, 250, 250), -1)
        comp = cv2.addWeighted(banner, 0.9, comp, 0.1, 0)
        cosv = "" if pd.isna(R.cos_to_flow) else f"  cos={R.cos_to_flow:+.2f}"
        txt = (f"R{int(R.rider_id)}  {verdict}{cosv}  disp={R.disp_px:.0f}px   "
               f"{first.img} -> {last.img}   pink=movement  orange=reference flow")
        cv2.putText(comp, txt, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
        tag = "WW" if ww is True else ("CROSS" if verdict == "crossing" else "OK")
        cv2.imwrite(str(out_dir / f"{tag}_R{int(R.rider_id):03d}.jpg"), comp)
        n += 1
    return n


def render_visualizations(det, riders, cfg, img_lookup, viz_dir, person_det=None):
    """One annotated JPG per capture that contains a kept detection:
    ROI overlay + flow arrow + bbox + rider id / facility / direction."""
    import cv2
    viz_dir.mkdir(parents=True, exist_ok=True)
    info = riders.set_index("rider_id")
    n = 0
    import pandas as pd
    if person_det is None:
        person_det = pd.DataFrame()
    all_imgs = sorted(set(det["img"]) | (set(person_det["img"]) if len(person_det) else set()))
    for img_name in all_imgs:
        g = det[det["img"] == img_name] if len(det) else det
        src = img_lookup.get(img_name)
        if src is None:
            continue
        img, _ = read_image(src)
        if img is None:
            continue
        overlay = img.copy()
        for rt, polys in cfg.rois.items():
            col = VIZ_COLORS.get(rt)
            if not col:
                continue
            for poly in polys:
                if len(poly) < 3:
                    continue
                pts = np.array([[int(x), int(y)] for x, y in poly], np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], col)
                cv2.polylines(img, [pts], True, col, 2)
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
        if cfg.flow and cfg.flow.get("vector"):
            v = cfg.flow["vector"]
            tip = (int(v["x2"]), int(v["y2"]))
            cv2.arrowedLine(img, (int(v["x1"]), int(v["y1"])), tip,
                            (0, 165, 255), 4, tipLength=0.06)
            for th, col in ((6, (0, 0, 0)), (2, (0, 165, 255))):
                cv2.putText(img, "FLOW", (tip[0] - 40, tip[1] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, th, cv2.LINE_AA)
        for _, r in g.iterrows():
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 3)
            ri = info.loc[int(r.rider_id)]
            label = f"R{int(r.rider_id)} {r.space_label} {ri.direction_displacement}"
            if ri.wrong_way_displacement is True:
                label += " WW!"
            for th, col in ((5, (0, 0, 0)), (2, (255, 255, 255))):
                cv2.putText(img, label, (x1, max(24, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, th, cv2.LINE_AA)
        if len(person_det):
            pd_img = person_det[person_det["img"] == img_name]
            if "space_label" in pd_img.columns:
                pd_img = pd_img[pd_img["space_label"].isin(KEEP_LABELS)]  # viz: in-ROI only
            for _, r in pd_img.iterrows():
                x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 220, 0), 2)
                for th, col in ((5, (0, 0, 0)), (2, (255, 255, 0))):
                    cv2.putText(img, f"person? {r.score:.2f}", (x1, max(24, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)
        cv2.imwrite(str(viz_dir / f"{Path(img_name).stem}_viz.jpg"), img)
        n += 1
    return n


def find_img_dir(data_root: Path, loc_id: str):
    num = loc_id.split("_", 1)[1]                     # "04-2"
    plain = num.lstrip("0") or "0"                    # "4-2"
    for cand in (f"Loc_{num}", f"Loc_{plain}", loc_id, loc_id.capitalize()):
        for sub in ("Bicyclist", "bicyclist", ""):
            p = data_root / cand / sub if sub else data_root / cand
            if p.is_dir():
                return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loc-id")
    ap.add_argument("--img-dir")
    ap.add_argument("--roi-json")
    ap.add_argument("--outdir")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--data-root")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640, help="inference size; 1280 helps small/night riders")
    ap.add_argument("--classes", type=int, nargs="+", default=[1])
    ap.add_argument("--nms-iou", type=float, default=0.70)
    ap.add_argument("--iomin-thr", type=float, default=0.60,
                    help="suppress same-frame box contained in a stronger box beyond this ratio")
    ap.add_argument("--assoc-gap", type=int, default=3)
    ap.add_argument("--min-move-px", type=float, default=25.0)
    ap.add_argument("--cos-gate", type=float, default=0.5,
                    help="|cos| below this counts as crossing (WW undefined)")
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--no-viz", dest="save_viz", action="store_false",
                    help="skip annotated visualization export (on by default)")
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--configs-dir", help="ROI config folder (default configs/locations)")
    args = ap.parse_args()
    args.classes = set(args.classes)

    cfg_dir = Path(args.configs_dir) if args.configs_dir else REPO_ROOT / "configs" / "locations_new"
    out_root = REPO_ROOT / "outputs_new"

    if args.batch:
        if not args.data_root:
            ap.error("--batch requires --data-root")
        root = Path(args.data_root)
        summaries = []
        for cfg in sorted(cfg_dir.glob("loc_*.json")):
            if "old" in cfg.stem.lower():
                continue
            loc = cfg.stem
            img_dir = find_img_dir(root, loc)
            if img_dir is None:
                print(f"[{loc}] image folder not found under {root}, skipped")
                continue
            s = run_location(loc, img_dir, cfg, out_root / loc, args)
            if s:
                summaries.append(s)
        pd.DataFrame(summaries).to_csv(out_root / "all_locations_summary.csv", index=False)
        print(f"\nBatch done: {len(summaries)} locations -> {out_root/'all_locations_summary.csv'}")
    else:
        if not (args.loc_id and args.img_dir):
            ap.error("need --loc-id and --img-dir (or --batch --data-root)")
        roi_json = Path(args.roi_json) if args.roi_json else cfg_dir / f"{args.loc_id}.json"
        outdir = Path(args.outdir) if args.outdir else out_root / args.loc_id
        run_location(args.loc_id, args.img_dir, roi_json, outdir, args)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Scene report: publication-style tables & figures
# ---------------------------------------------------------------------------
FACILITY_COLORS = {"sidewalk": "#2a78d6", "bike_lane": "#1baf7a",
                   "roadway": "#d62728", "crosswalk": "#c05ec4"}
DIRECTION_COLORS = {"along_flow": "#1baf7a", "against_flow": "#d62728",
                    "cross_flow": "#2a78d6", "unknown": "#9a9890"}
_INK, _INK2, _GRID = "#0b0b0b", "#52514e", "#e5e5e2"


def _mpl_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": _GRID, "axes.labelcolor": _INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": _GRID, "grid.linewidth": 0.8,
        "xtick.color": _INK2, "ytick.color": _INK2,
        "text.color": _INK, "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlecolor": _INK, "figure.dpi": 110, "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })



def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    h = z * ((ph * (1 - ph) + z * z / (4 * n)) / n) ** 0.5
    return ((c - h) / d, (c + h) / d)


def scene_stats_table(riders, loc_id="", manual=None, n_person_candidates=None):
    """Headline stats for one scene; optional manual=(along, against) benchmark."""
    import pandas as pd
    n = len(riders)
    dd = riders["direction_displacement"]
    along = int((dd == "along_flow").sum())
    against = int((dd == "against_flow").sum())
    cross = int((dd == "cross_flow").sum())
    unknown = n - along - against - cross
    dk = along + against
    sw, bl, rd = (int(riders[c].sum()) for c in
                  ("in_sidewalk_any", "in_bike_lane_any", "in_roadway_any"))

    def pct(k, d):
        return f"{k/d:.1%}" if d else "-"

    def ww_cell(a, g):
        d = a + g
        if not d:
            return "-"
        lo, hi = _wilson(g, d)
        return f"{g/d:.1%}  (95% CI {lo:.0%}-{hi:.0%})"

    rows = [
        ("Riders counted",              f"{n}",                        ""),
        ("Direction known",             f"{dk}  ({pct(dk, n)})",       ""),
        ("  along flow",                f"{along}",                    ""),
        ("  against flow (wrong-way)",  f"{against}",                  ""),
        ("Wrong-way rate",              ww_cell(along, against),       ""),
        ("Crossing (perpendicular)",    f"{cross}",                    ""),
        ("No direction (single frame / stationary)", f"{unknown}",     ""),
        ("Sidewalk involvement",        f"{sw}  ({pct(sw, n)})",       ""),
        ("Bike lane involvement",       f"{bl}  ({pct(bl, n)})",       ""),
        ("Roadway involvement",         f"{rd}  ({pct(rd, n)})",       ""),
    ]
    if n_person_candidates is not None:
        rows.append(("Person-only candidates (possible missed riders)",
                     f"{n_person_candidates}", ""))
    cols = ["Metric", "Pipeline", "Manual"]
    if manual is not None:
        ma, mg = manual
        md = ma + mg
        man = {"Riders counted": f"{md}",
               "Direction known": f"{md}  (100%)",
               "  along flow": f"{ma}",
               "  against flow (wrong-way)": f"{mg}",
               "Wrong-way rate": ww_cell(ma, mg)}
        rows = [(m, v, man.get(m, "")) for m, v, _ in rows]
    else:
        cols = cols[:2]
        rows = [r[:2] for r in rows]
    df = pd.DataFrame(rows, columns=cols)
    sty = (df.style.hide(axis="index")
           .set_caption(f"Scene summary — {loc_id}" + ("  (Manual = human count benchmark)" if manual else ""))
           .map(lambda v: "background-color: #fdecea; font-weight: 700;" if isinstance(v, str) and "CI" in v else "",
                subset=cols[1:])
           .set_properties(subset=cols[1:], **{"text-align": "right", "font-variant-numeric": "tabular-nums"})
           .set_table_styles([
               {"selector": "caption", "props": f"caption-side: top; font-weight: 600; color: {_INK}; padding: 6px 0;"},
               {"selector": "th", "props": f"text-align: left; color: {_INK2}; border-bottom: 1.5px solid {_INK}; padding: 4px 12px;"},
               {"selector": "td", "props": f"padding: 4px 12px; border-bottom: 0.5px solid {_GRID};"},
           ]))
    return sty


def qc_funnel_table(summary):
    """Detection funnel with per-stage attrition — the QC record for defense."""
    import pandas as pd
    f = summary["funnel"]
    rows = [
        ("Captures on disk",            f["captures"],                "-"),
        ("  unreadable files",          -f["unreadable"],             "corrupt beyond recovery, skipped"),
        ("  truncated but recovered",   f["truncated_recovered"],     "decoded with tolerant reader (kept)"),
        ("Raw bicycle detections",      f["detections_raw"],          f"YOLO {summary['params']['model']} conf>={summary['params']['conf']}"),
        ("After NMS",                   f["after_nms"],               f"same-frame IoU >= {summary['params']['nms_iou']} removed"),
        ("After containment dedup",     f["after_containment_dedup"], "nested duplicate boxes removed"),
        ("Inside study ROI",            f["in_roi"],                  "bottom-edge vote in annotated facilities"),
        ("Riders (counting unit)",      f["riders"],                  "nearby-capture association"),
        ("Person-only candidates",      f["person_candidates"],       "possible missed riders, flagged for review"),
    ]
    df = pd.DataFrame(rows, columns=["Stage", "Count", "Rule"])
    sty = (df.style.hide(axis="index")
           .set_caption(f"Detection funnel — {summary['location_id']}")
           .set_properties(subset=["Count"], **{"text-align": "right", "font-variant-numeric": "tabular-nums"})
           .set_properties(subset=["Rule"], **{"color": _INK2, "font-size": "0.9em"})
           .set_table_styles([
               {"selector": "caption", "props": f"caption-side: top; font-weight: 600; color: {_INK}; padding: 6px 0;"},
               {"selector": "th", "props": f"text-align: left; color: {_INK2}; border-bottom: 1.5px solid {_INK}; padding: 4px 12px;"},
               {"selector": "td", "props": f"padding: 4px 12px; border-bottom: 0.5px solid {_GRID};"},
           ]))
    return sty


def riders_table(riders):
    """Riders styled for reading: check marks, colored direction, tabular numbers."""
    import pandas as pd
    d = riders.copy()
    for c in ("in_sidewalk_any", "in_bike_lane_any", "in_roadway_any"):
        d[c] = d[c].map({True: "\u2713", False: ""})
    d = d[["rider_id", "n_obs", "img_first", "img_last",
           "in_sidewalk_any", "in_bike_lane_any", "in_roadway_any",
           "disp_px", "direction_displacement", "cos_to_flow", "wrong_way_displacement"]]
    d.columns = ["rider", "obs", "first capture", "last capture",
                 "sidewalk", "bike lane", "roadway",
                 "disp (px)", "direction", "cos", "wrong-way"]
    def dir_color(v):
        return f"color: {DIRECTION_COLORS.get(v, _INK)}; font-weight: 600;"
    sty = (d.style.hide(axis="index")
           .set_caption(f"Riders — one row per counted rider (n={len(d)})")
           .format({"disp (px)": "{:.0f}", "cos": lambda v: "" if pd.isna(v) else f"{v:+.2f}",
                    "wrong-way": lambda v: "" if pd.isna(v) else ("YES" if v else "no")}, na_rep="")
           .map(dir_color, subset=["direction"])
           .set_properties(subset=["sidewalk"], **{"color": FACILITY_COLORS["sidewalk"], "text-align": "center"})
           .set_properties(subset=["bike lane"], **{"color": FACILITY_COLORS["bike_lane"], "text-align": "center"})
           .set_properties(subset=["roadway"], **{"color": FACILITY_COLORS["roadway"], "text-align": "center"})
           .set_properties(subset=["rider", "obs", "disp (px)", "cos"], **{"text-align": "right", "font-variant-numeric": "tabular-nums"})
           .set_table_styles([
               {"selector": "caption", "props": f"caption-side: top; font-weight: 600; color: {_INK}; padding: 6px 0;"},
               {"selector": "th", "props": f"text-align: left; color: {_INK2}; border-bottom: 1.5px solid {_INK}; padding: 4px 10px;"},
               {"selector": "td", "props": f"padding: 3px 10px; border-bottom: 0.5px solid {_GRID};"},
           ]))
    return sty


def fig_facility(riders, loc_id, save_to=None):
    import matplotlib.pyplot as plt
    _mpl_style()
    n = len(riders)
    vals = [int(riders["in_sidewalk_any"].sum()),
            int(riders["in_bike_lane_any"].sum()),
            int(riders["in_roadway_any"].sum())]
    labels = ["Sidewalk", "Bike lane", "Roadway"]
    colors = [FACILITY_COLORS["sidewalk"], FACILITY_COLORS["bike_lane"], FACILITY_COLORS["roadway"]]
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.55)
    for b, v in zip(bars, vals[::-1]):
        share = v / n if n else 0
        ax.text(b.get_width() + max(vals + [1]) * 0.02, b.get_y() + b.get_height() / 2,
                f"{v}  ({share:.0%})", va="center", color=_INK, fontweight="bold")
    ax.set_xlim(0, max(vals + [1]) * 1.22)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"Facility involvement — {loc_id}  (n={n} riders, not mutually exclusive)")
    ax.grid(axis="y", visible=False)
    if save_to:
        fig.savefig(save_to)
    return fig


def fig_confidence(det, person, loc_id, save_to=None):
    import matplotlib.pyplot as plt
    import numpy as np
    _mpl_style()
    fig, ax = plt.subplots(figsize=(6.5, 3))
    bins = np.arange(0.10, 1.01, 0.05)
    if person is not None and len(person):
        ax.hist(person["score"], bins=bins, color="#c9c8c2", label=f"person candidates (n={len(person)})")
    if len(det):
        ax.hist(det["score"], bins=bins, color="#2a78d6", label=f"bicycle detections (n={len(det)})")
    ax.set_xlabel("detection confidence")
    ax.set_ylabel("count")
    ax.set_title(f"Detector confidence — {loc_id}")
    ax.legend(frameon=False)
    if save_to:
        fig.savefig(save_to)
    return fig


def fig_spatial(det, person, cfg, sample_image, loc_id, save_to=None):
    """Spatial footprint: ROI outlines + bbox bottom-center of every detection."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.lines import Line2D
    import numpy as np
    _mpl_style()
    fig, ax = plt.subplots(figsize=(9, 9 * cfg.h / cfg.w))
    if sample_image is not None:
        img, _ = read_image(sample_image)
        if img is not None:
            gray = img[:, :, ::-1].mean(axis=2)
            ax.imshow(gray, cmap="gray", alpha=0.55, extent=[0, cfg.w, cfg.h, 0])
    for rt, polys in cfg.rois.items():
        col = FACILITY_COLORS.get(rt)
        if not col:
            continue
        for poly in polys:
            if len(poly) >= 3:
                ax.add_patch(MplPoly(np.array(poly), closed=True, fill=True,
                                     facecolor=col, alpha=0.10, edgecolor=col, linewidth=1.8))
    if cfg.flow and cfg.flow.get("vector"):
        v = cfg.flow["vector"]
        ax.annotate("", xy=(v["x2"], v["y2"]), xytext=(v["x1"], v["y1"]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#eda100"))
    if person is not None and len(person):
        pin = person[person["space_label"].isin(KEEP_LABELS)] if "space_label" in person.columns else person
        ax.scatter((pin.x1 + pin.x2) / 2, pin.y2, s=46, marker="o", facecolor="none",
                   edgecolor="#9a9890", linewidth=1.6, label="person candidate")
    for lab in ["sidewalk", "bike_lane", "roadway", "crosswalk"]:
        dd = det[det["space_label"] == lab] if len(det) else det
        if len(dd):
            ax.scatter((dd.x1 + dd.x2) / 2, dd.y2, s=60, color=FACILITY_COLORS[lab],
                       edgecolor="white", linewidth=1.2, label=f"rider on {lab.replace('_', ' ')}")
    ax.set_xlim(0, cfg.w); ax.set_ylim(cfg.h, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"Spatial footprint — {loc_id} (marker = bbox bottom center)")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    if save_to:
        fig.savefig(save_to)
    return fig


def fig_contact_sheet(viz_dir, loc_id, max_n=8, save_to=None):
    import matplotlib.pyplot as plt
    import cv2
    _mpl_style()
    files = sorted(Path(viz_dir).glob("*_viz.jpg"))[:max_n]
    if not files:
        return None
    cols = 2
    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.8 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in axes[len(files):]:
        ax.axis("off")
    for ax, f in zip(axes, files):
        im = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        ax.imshow(im); ax.axis("off")
        ax.set_title(f.name.replace("_viz.jpg", ""), fontsize=9, color=_INK2)
    fig.suptitle(f"Annotated captures — {loc_id}", fontweight="bold", y=1.0)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to)
    return fig


def person_review_table(person):
    """Review queue: person-only frames, highest confidence first."""
    import pandas as pd
    q = person[~person["frame_has_bicycle"]].copy() if "frame_has_bicycle" in person.columns else person.copy()
    if not len(q):
        return None
    q["in_study_roi"] = q["space_label"].isin(KEEP_LABELS) if "space_label" in q.columns else False
    q = q.sort_values(["in_study_roi", "score"], ascending=[False, False])
    q = q[["img", "score", "space_label", "in_study_roi"]]
    q.columns = ["capture", "confidence", "ROI label", "in study ROI"]
    sty = (q.style.hide(axis="index")
           .set_caption(f"Person-only candidates — review queue (n={len(q)}; confirm rider vs pedestrian)")
           .format({"confidence": "{:.2f}"})
           .map(lambda v: "color: #1baf7a; font-weight: 600;" if v is True else ("color: #9a9890;" if v is False else ""),
                subset=["in study ROI"])
           .set_table_styles([
               {"selector": "caption", "props": f"caption-side: top; font-weight: 600; color: {_INK}; padding: 6px 0;"},
               {"selector": "th", "props": f"text-align: left; color: {_INK2}; border-bottom: 1.5px solid {_INK}; padding: 4px 10px;"},
               {"selector": "td", "props": f"padding: 3px 10px; border-bottom: 0.5px solid {_GRID};"},
           ]))
    return sty


def direction_table(riders):
    """Every direction-judged rider with its frames — the WW verification list."""
    import pandas as pd
    d = riders[riders["direction_displacement"].isin(
        ["along_flow", "against_flow", "cross_flow"])].copy()
    if not len(d):
        return None
    order = {"against_flow": 0, "cross_flow": 1, "along_flow": 2}
    d = d.sort_values(by="direction_displacement", key=lambda c: c.map(order))
    d["check image"] = d.apply(lambda r: ("WW" if r.wrong_way_displacement is True or r.wrong_way_displacement == True
                                          else ("CROSS" if r.direction_displacement == "cross_flow" else "OK"))
                                          + f"_R{int(r.rider_id):03d}.jpg", axis=1)
    d = d[["rider_id", "img_first", "img_last", "disp_px", "cos_to_flow",
           "direction_displacement", "wrong_way_displacement", "check image"]]
    d.columns = ["rider", "first frame", "last frame", "disp (px)", "cos",
                 "direction", "wrong-way", "check image"]
    def dir_color(v):
        return f"color: {DIRECTION_COLORS.get(v, _INK)}; font-weight: 700;"
    sty = (d.style.hide(axis="index")
           .set_caption(f"Direction verdicts — verify each against direction_check/ (n={len(d)})")
           .format({"disp (px)": "{:.0f}", "cos": "{:+.2f}",
                    "wrong-way": lambda v: "YES" if v is True or v == True else "no"}, na_rep="")
           .map(dir_color, subset=["direction"])
           .map(lambda v: "background-color: #fdecea; font-weight: 700;" if v == "YES" else "",
                subset=["wrong-way"])
           .set_table_styles([
               {"selector": "caption", "props": f"caption-side: top; font-weight: 600; color: {_INK}; padding: 6px 0;"},
               {"selector": "th", "props": f"text-align: left; color: {_INK2}; border-bottom: 1.5px solid {_INK}; padding: 4px 10px;"},
               {"selector": "td", "props": f"padding: 3px 10px; border-bottom: 0.5px solid {_GRID};"},
           ]))
    return sty


def generate_report(outdir, roi_json, loc_id, show=True, manual=None):
    """Build all report tables & figures for one location run.
    Figures are also saved to outdir/report/ as 200-dpi PNGs for the paper."""
    import json as _json
    import pandas as pd
    outdir = Path(outdir)
    rep_dir = outdir / "report"
    rep_dir.mkdir(exist_ok=True)
    summary = _json.loads((outdir / "scene_summary.json").read_text())
    riders = pd.read_csv(outdir / "riders.csv") if (outdir / "riders.csv").exists() else pd.DataFrame()
    det = pd.read_csv(outdir / "detections_riders.csv") if (outdir / "detections_riders.csv").exists() else pd.DataFrame()
    person = pd.read_csv(outdir / "person_candidates.csv") if (outdir / "person_candidates.csv").exists() else pd.DataFrame()
    from src.scene.roi_new import load_roi_config
    cfg = load_roi_config(roi_json)

    out = {"summary": summary}
    if len(riders):
        out["stats"] = scene_stats_table(
            riders, loc_id, manual=manual,
            n_person_candidates=summary.get("funnel", {}).get("person_candidates"))
    out["funnel"] = qc_funnel_table(summary)
    if len(riders):
        out["riders_table"] = riders_table(riders)
        dt = direction_table(riders)
        if dt is not None:
            out["direction_table"] = dt
        out["fig_facility"] = fig_facility(riders, loc_id, save_to=rep_dir / "facility_involvement.png")
    if len(person):
        prt = person_review_table(person)
        if prt is not None:
            out["person_review"] = prt
    if len(det) or len(person):
        out["fig_confidence"] = fig_confidence(det, person, loc_id, save_to=rep_dir / "detector_confidence.png")
        sample = None
        vd = sorted((outdir / "viz").glob("*_viz.jpg")) if (outdir / "viz").exists() else []
        out["fig_spatial"] = fig_spatial(det, person, cfg, None, loc_id, save_to=rep_dir / "spatial_footprint.png")
        out["fig_contact"] = fig_contact_sheet(outdir / "viz", loc_id, save_to=rep_dir / "contact_sheet.png")
    return out

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
    pd.DataFrame(integrity).to_csv(outdir / "integrity_report.csv", index=False)

    if len(det) == 0:
        summary = {"location_id": loc_id, "n_captures": len(img_paths), "n_riders": 0}
        (outdir / "scene_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{loc_id}] no detections")
        return summary

    det = nms_lite_per_frame(det, frame_col="img_num", score_col="score", iou_thr=getattr(args, "nms_iou", 0.70))

    thr = FrameLabelThresholds(bottom_edge_npts=5, vote_min_pts=2)
    det["space_label"] = det.apply(
        lambda r: frame_space_label((r.x1, r.y1, r.x2, r.y2), roi, thr), axis=1)
    det = det[det["space_label"].isin(KEEP_LABELS)].copy()

    assoc = AssocParams(max_frame_gap=getattr(args, "assoc_gap", 3),
                        min_move_px=getattr(args, "min_move_px", 8.0),
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
        "params": {
            "model": args.model, "conf": args.conf, "classes": sorted(args.classes),
            "nms_iou": getattr(args, "nms_iou", 0.70), "assoc_max_frame_gap": getattr(args, "assoc_gap", 3),
            "min_move_px": getattr(args, "min_move_px", 8.0), "cos_gate": getattr(args, "cos_gate", 0.5), "roi_exclusive": True,
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
            cv2.arrowedLine(img, (int(v["x1"]), int(v["y1"])), (int(v["x2"]), int(v["y2"])),
                            (0, 165, 255), 4, tipLength=0.25)
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
    ap.add_argument("--assoc-gap", type=int, default=3)
    ap.add_argument("--min-move-px", type=float, default=8.0)
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

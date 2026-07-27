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
    riders.csv           one row per rider (counting unit), with dominant_space
    stationary_objects.csv detections removed as parked bikes (audit trail:
                         verify none of them is a real rider)
    scene_summary.json   counts + parameters
    integrity_report.csv unreadable / truncated captures
    crops/               (--save-crops) one crop per appearance, for
                         orientation labeling

Association uses image-number gaps AND EXIF capture times: motion-triggered
cameras make consecutive image numbers minutes apart, so a time gap above
--max-time-gap-s always breaks the chain (prevents merging two different
riders into one).
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
    rescue_min_app_sim: float = 0.30 # appearance correlation required when fingerprints exist
                                     # (no size gate: depth motion legitimately changes bbox area >10x)
    trust_rescue_direction: bool = False  # queued riders make rescued pairs point backwards
                                          # (validated at loc_04/04-2: rescue WW 93-94% vs
                                          # manual ~26-28%); rescue merges count riders but
                                          # never contribute a displacement direction
    trust_rescue_pair_s: float = 4.0      # EXCEPTION: a rescued chain whose EXIF span is
                                          # within this window is the camera's documented
                                          # double-shot (two images ~2s apart) of one fast
                                          # rider — its displacement is real. Direction is
                                          # trusted and tagged direction_source="pair_rescue"
                                          # so it can be validated separately. 0 disables.
    min_move_px: float = 25.0     # calibrated: stationary-target jitter is <13px, real riders move >50px between captures
    cos_gate: float = 0.5         # |cos| below this = crossing, not along/against
    max_time_gap_s: float = 30.0  # motion-triggered cameras: consecutive image numbers can
                                  # be minutes apart, so a frame-gap check alone merges two
                                  # different people. EXIF capture times farther apart than
                                  # this always break the association (0 disables).


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
    last_app: Dict[int, object] = {}
    rescued_det_keys = set()
    rescue_flags: List[bool] = []
    # rid -> (last_img_num, last_bbox, last_ts)
    active: Dict[int, Tuple[int, BBox, Optional[float]]] = {}
    rider_ids: List[int] = []
    max_dt = float(getattr(params, "max_time_gap_s", 30.0) or 0)

    for img_num, frame in g.groupby("img_num", sort=True):
        rows = list(frame.itertuples())
        ts_now = getattr(rows[0], "ts", None)
        ts_now = float(ts_now) if ts_now is not None and pd.notna(ts_now) else None

        # expire riders that are too far in the past (image number, or real
        # capture time when EXIF is available)
        for rid in [r for r, (n, _, t) in active.items()
                    if img_num - n > params.max_frame_gap
                    or (max_dt > 0 and t is not None and ts_now is not None
                        and ts_now - t > max_dt)]:
            del active[rid]

        bbs = [(float(r.x1), float(r.y1), float(r.x2), float(r.y2)) for r in rows]

        # greedy one-to-one matching: closest pairs first
        cands = []
        for i, bb in enumerate(bbs):
            c = _bottom_center(bb)
            gate = max(params.min_gate_px, params.max_dist_factor * _diag(bb))
            for rid, (_, last_bb, _t) in active.items():
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
            recent = [rid for rid, (n, _, _t) in active.items()
                      if img_num - n <= params.rescue_gap and rid not in used_rids]
            if len(recent) == 1:
                rid0 = recent[0]
                bb0, bb1 = active[rid0][1], bbs[0]
                c = _bottom_center(bb1)
                lc = _bottom_center(bb0)
                sim = _app_sim(last_app.get(rid0), rows[0].app if hasattr(rows[0], "app") else None)
                if (abs(c[0] - lc[0]) <= params.rescue_max_frac * frame_width
                        and (sim is None or sim >= params.rescue_min_app_sim)):
                    assigned_det[0] = rid0
                    rescued_det_keys.add((int(img_num), 0))

        for i, bb in enumerate(bbs):
            rid = assigned_det.get(i)
            if rid is None:
                rid = next_rid
                next_rid += 1
            rider_ids.append(rid)
            rescue_flags.append((int(img_num), i) in rescued_det_keys)
            active[rid] = (int(img_num), bb, ts_now)
            if hasattr(rows[i], "app"):
                last_app[rid] = rows[i].app

    g["rider_id"] = rider_ids
    g["assoc_rescue"] = rescue_flags
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
    cols = ["rider_id", "n_obs", "assoc_rescue", "img_first", "img_last",
            "img_num_first", "img_num_last", "in_sidewalk_any", "in_bike_lane_any",
            "in_roadway_any", "dominant_space", "dwell_s", "disp_px",
            "direction_displacement", "direction_source", "cos_to_flow",
            "wrong_way_displacement", "orientation"]
    if det_df is None or len(det_df) == 0:
        return pd.DataFrame(columns=cols)

    prio = ("crosswalk", "bike_lane", "sidewalk", "roadway")
    rows = []
    for rid, g in det_df.groupby("rider_id", sort=True):
        g = g.sort_values("img_num")
        labels = g["space_label"].tolist()

        # dominant facility: majority vote over the rider's observations,
        # ties broken by priority — same definition the manual count uses
        # ("where did this person mainly ride"), unlike the any-involvement
        # flags below which light up on a single touching frame
        cnt = {k: labels.count(k) for k in prio}
        best = max(prio, key=lambda k: (cnt[k], -prio.index(k)))
        dominant = best if cnt[best] > 0 else "unknown"
        if dominant == "crosswalk":
            dominant = "roadway"   # crossing is roadway semantics

        ts_vals = ([float(v) for v in g["ts"].tolist() if pd.notna(v)]
                   if "ts" in g.columns else [])
        dwell = (max(ts_vals) - min(ts_vals)) if len(ts_vals) >= 2 else None

        first, last = g.iloc[0], g.iloc[-1]
        bc0 = _bottom_center((first.x1, first.y1, first.x2, first.y2))
        bc1 = _bottom_center((last.x1, last.y1, last.x2, last.y2))
        dx, dy = bc1[0] - bc0[0], bc1[1] - bc0[1]
        disp = (dx * dx + dy * dy) ** 0.5

        direction = "unknown"
        cos = None
        wrong_way: Optional[bool] = None
        rescued = bool(g["assoc_rescue"].any()) if "assoc_rescue" in g.columns else False
        # rescue pairs are distrusted (queued riders point backwards), EXCEPT
        # when the whole chain sits inside the camera's ~2s double-shot window:
        # that is one fast rider's genuine displacement
        pair_ok = (rescued and dwell is not None
                   and getattr(params, "trust_rescue_pair_s", 4.0) > 0
                   and dwell <= getattr(params, "trust_rescue_pair_s", 4.0))
        direction_trusted = params.trust_rescue_direction or not rescued or pair_ok
        direction_source = None
        if flow is not None and len(g) >= 2 and disp >= params.min_move_px and direction_trusted:
            direction_source = "pair_rescue" if (rescued and pair_ok) else "gated"
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
            "assoc_rescue": bool(g["assoc_rescue"].any()) if "assoc_rescue" in g.columns else False,
            "img_first": first.img,
            "img_last": last.img,
            "img_num_first": int(first.img_num),
            "img_num_last": int(last.img_num),
            "in_sidewalk_any": "sidewalk" in labels,
            "in_bike_lane_any": "bike_lane" in labels,
            "in_roadway_any": ("roadway" in labels) or ("crosswalk" in labels),
            "dominant_space": dominant,
            "dwell_s": None if dwell is None else round(dwell, 1),
            "disp_px": round(disp, 1),
            "direction_displacement": direction,
            "direction_source": direction_source,
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


def _app_hist(img, x1, y1, x2, y2):
    """Compact HS color histogram of a crop — appearance fingerprint."""
    import cv2
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([crop], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().tolist()


def _app_sim(a, b):
    """Correlation of two fingerprints; None when either is missing."""
    import numpy as np
    if a is None or b is None or isinstance(a, float) or isinstance(b, float):
        return None
    try:
        import cv2
        return float(cv2.compareHist(np.array(a, np.float32), np.array(b, np.float32),
                                     cv2.HISTCMP_CORREL))
    except Exception:
        return None


def read_exif_ts(path):
    """EXIF capture time as epoch seconds (DateTimeOriginal, falling back to
    DateTime). None when the file has no usable timestamp — everything that
    consumes ts degrades gracefully to image-number logic in that case."""
    try:
        from PIL import Image
        from datetime import datetime
        with Image.open(path) as im:
            ex = im.getexif()
            v = ex.get(306)                       # DateTime
            try:
                v = ex.get_ifd(0x8769).get(36867) or v  # DateTimeOriginal
            except Exception:
                pass
        if not v:
            return None
        return datetime.strptime(str(v).strip()[:19], "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        return None


def flag_stationary_detections(det, radius_px=30.0, min_hits=6,
                               min_span_frames=50, min_span_s=300.0,
                               min_density=0.6):
    """Flag detections of parked bicycles.

    A parked bike fires a detection at the same spot every time a passer-by
    triggers the camera; the frame-gap association then splits it into many
    fake "riders" (loc_15 counted 122% of the manual benchmark this way).
    Greedy centroid clustering of bbox bottom-centers: a cluster hit in
    >= min_hits distinct captures whose span exceeds min_span_s EXIF seconds
    (or min_span_frames image numbers when no timestamps exist) cannot be one
    passing rider — a cyclist waiting at a light clears in well under 5
    minutes.

    min_density separates a parked object from a busy chokepoint. An object
    physically present in the scene appears in (nearly) every capture inside
    its time window — whoever trips the shutter, the parked bike is in view.
    Hundreds of DIFFERENT riders passing one spot of a busy bike lane also
    build a long-span cluster (loc_17: 413 riders through one lane), but each
    appears in only a scattered fraction of the window's captures. Clusters
    hitting fewer than min_density of the window's captures are kept as
    riders. Returns a boolean Series aligned with det.index.
    """
    flags = pd.Series(False, index=det.index)
    if det is None or len(det) == 0:
        return flags
    all_frames = np.array(sorted(pd.unique(det["img_num"])))
    have_ts = "ts" in det.columns
    clusters = []  # [cx, cy, n, frame_set, ts_list, idx_list]
    for idx, r in det.sort_values("img_num").iterrows():
        cx, cy = (float(r.x1) + float(r.x2)) / 2.0, float(r.y2)
        best, bd = None, None
        for c in clusters:
            d = ((cx - c[0]) ** 2 + (cy - c[1]) ** 2) ** 0.5
            if d <= radius_px and (bd is None or d < bd):
                best, bd = c, d
        if best is None:
            best = [cx, cy, 1, {int(r.img_num)}, [], [idx]]
            clusters.append(best)
        else:
            best[0] = (best[0] * best[2] + cx) / (best[2] + 1)
            best[1] = (best[1] * best[2] + cy) / (best[2] + 1)
            best[2] += 1
            best[3].add(int(r.img_num))
            best[5].append(idx)
        if have_ts and pd.notna(r.ts):
            best[4].append(float(r.ts))
    for c in clusters:
        if len(c[3]) < min_hits:
            continue
        if len(c[4]) >= 2:
            span_ok = (max(c[4]) - min(c[4])) >= min_span_s
        else:
            span_ok = (max(c[3]) - min(c[3])) >= min_span_frames
        if not span_ok:
            continue
        lo, hi = min(c[3]), max(c[3])
        n_window = int(((all_frames >= lo) & (all_frames <= hi)).sum())
        if len(c[3]) / max(n_window, 1) < min_density:
            continue   # busy chokepoint: many riders share the spot, none stays
        flags.loc[c[5]] = True
    return flags


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
    import shutil
    for sub in ("crops", "crops_person", "viz", "direction_check"):
        d = outdir / sub
        if d.exists():
            shutil.rmtree(d)

    cfg = load_roi_config(roi_json)
    roi = ExclusiveROIMaskEngine(cfg)
    flow = roi.flow_vector()

    img_paths = collect_images(Path(img_dir), getattr(args, "max_images", None))
    if not img_paths:
        print(f"[{loc_id}] no images in {img_dir}")
        return None
    print(f"[{loc_id}] {len(img_paths)} captures, flow={'yes' if flow else 'none'}")

    raw_csv = outdir / "detections_raw.csv"
    reuse = bool(getattr(args, "reuse_detections", False)) and raw_csv.exists()
    if reuse:
        # detector output is unchanged — only post-processing differs, so skip
        # the (hours-long) YOLO pass and replay from the saved raw detections
        import ast
        det = pd.read_csv(raw_csv)
        if "app" in det.columns:
            det["app"] = det["app"].apply(
                lambda v: ast.literal_eval(v) if isinstance(v, str) else None)
        if "ts" not in det.columns:
            det["ts"] = None
        try:
            integrity = pd.read_csv(outdir / "integrity_report.csv").to_dict("records")
        except Exception:
            integrity = []
        person_rows = []
        n_with_ts = int(det.dropna(subset=["ts"])["img"].nunique()) if "ts" in det.columns else 0
        print(f"[{loc_id}] reusing detections_raw.csv ({len(det)} detections) — YOLO skipped")
        det_rows = None
    else:
        from ultralytics import YOLO
        model = YOLO(args.model)
        det_rows, person_rows, integrity = [], [], []
        n_with_ts = 0
        for i, p in enumerate(img_paths):
            img, status = read_image(p)
            if img is None:
                integrity.append({"img": p.name, "status": status})
                continue
            if status != "ok":
                integrity.append({"img": p.name, "status": status})
            ts = read_exif_ts(p)
            if ts is not None:
                n_with_ts += 1
            r = model(img, conf=args.conf, imgsz=getattr(args, "imgsz", 640), verbose=False)[0]
            if r.boxes is None:
                continue
            for (x1, y1, x2, y2), s, c in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
            ):
                row = {
                    "img": p.name, "img_num": imgnum(p.name), "frame_global": i, "ts": ts,
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "score": float(s), "cls": int(c),
                    "app": _app_hist(img, x1, y1, x2, y2),
                }
                if c in args.classes:
                    det_rows.append(row)
                elif c == 0 and getattr(args, "person_probe", True):
                    # person without a counted class: candidate for a missed rider
                    person_rows.append(row)
            if (i + 1) % 500 == 0:
                print(f"[{loc_id}]   {i+1}/{len(img_paths)} captures, {len(det_rows)} detections")

    if not reuse:
        det = pd.DataFrame(det_rows)
        det.to_csv(outdir / "detections_raw.csv", index=False)
        pd.DataFrame(integrity).to_csv(outdir / "integrity_report.csv", index=False)
    n_det_raw = len(det)

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

    n_stationary = 0
    if getattr(args, "stationary_filter", True) and len(det):
        stat_mask = flag_stationary_detections(
            det,
            radius_px=getattr(args, "stationary_radius", 30.0),
            min_hits=getattr(args, "stationary_hits", 6),
            min_span_frames=getattr(args, "stationary_span_frames", 50),
            min_span_s=getattr(args, "stationary_span_s", 300.0),
            min_density=getattr(args, "stationary_min_density", 0.6))
        n_stationary = int(stat_mask.sum())
        if n_stationary:
            det[stat_mask].to_csv(outdir / "stationary_objects.csv", index=False)
            print(f"[{loc_id}] stationary filter: {n_stationary} detections at parked-bike "
                  f"spots removed -> stationary_objects.csv (verify the crops there)")
        det = det[~stat_mask].copy()

    assoc = AssocParams(max_frame_gap=getattr(args, "assoc_gap", 3),
                        min_move_px=getattr(args, "min_move_px", 25.0),
                        cos_gate=getattr(args, "cos_gate", 0.5),
                        max_time_gap_s=getattr(args, "max_time_gap_s", 30.0),
                        trust_rescue_pair_s=getattr(args, "trust_rescue_pair_s", 4.0))
    det = associate_riders(det, assoc)
    det.to_csv(outdir / "detections_riders.csv", index=False)

    riders = summarize_riders(det, flow, assoc)
    riders.to_csv(outdir / "riders.csv", index=False)

    if reuse:
        pc_csv = outdir / "person_candidates.csv"
        try:
            person_df = pd.read_csv(pc_csv) if pc_csv.exists() else pd.DataFrame()
        except Exception:
            person_df = pd.DataFrame()
        n_person = len(person_df)
    else:
        person_df = pd.DataFrame(person_rows)
        n_person = len(person_rows)
    if len(person_df):
        person_df = nms_lite_per_frame(person_df, frame_col="img_num", score_col="score", iou_thr=0.55)
        person_df = dedup_contained_per_frame(person_df, iomin_thr=0.55)
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
        "n_person_candidates": int(n_person),
        "n_riders_multi_obs": int((riders["n_obs"] >= 2).sum()),
        "riders_in_sidewalk": int(riders["in_sidewalk_any"].sum()),
        "riders_in_bike_lane": int(riders["in_bike_lane_any"].sum()),
        "riders_in_roadway": int(riders["in_roadway_any"].sum()),
        "riders_dom_sidewalk": int((riders["dominant_space"] == "sidewalk").sum()),
        "riders_dom_bike_lane": int((riders["dominant_space"] == "bike_lane").sum()),
        "riders_dom_roadway": int((riders["dominant_space"] == "roadway").sum()),
        "direction_known_displacement": n_dir,
        "riders_crossing_flow": n_cross,
        "wrong_way_displacement": n_ww,
        # validation hook for the 2s-pair rescue exception: check this WW share
        # against the manual benchmark before relying on pair_rescue directions
        "direction_pair_rescue": int((riders["direction_source"] == "pair_rescue").sum()),
        "ww_pair_rescue": int(((riders["direction_source"] == "pair_rescue")
                               & (riders["wrong_way_displacement"] == True)).sum()),
        "note": "wrong_way is provisional (displacement only); single-obs riders await orientation labels",
        "funnel": {
            "captures": int(len(img_paths)),
            "unreadable": int(sum(1 for r in integrity if str(r["status"]).startswith("unreadable"))),
            "truncated_recovered": int(sum(1 for r in integrity if r["status"] == "truncated_recovered")),
            "detections_raw": int(n_det_raw),
            "after_nms": int(n_after_nms),
            "after_containment_dedup": int(n_after_dedup),
            "in_roi": int(n_in_roi),
            "stationary_object_dets": int(n_stationary),
            "riders": int(riders.shape[0]),
            "person_candidates": int(n_person),
            "captures_with_exif_ts": int(n_with_ts),
        },
        "params": {
            "model": args.model, "conf": args.conf, "imgsz": getattr(args, "imgsz", 640),
            "classes": sorted(args.classes),
            "nms_iou": getattr(args, "nms_iou", 0.70), "assoc_max_frame_gap": getattr(args, "assoc_gap", 3),
            "min_move_px": getattr(args, "min_move_px", 25.0), "cos_gate": getattr(args, "cos_gate", 0.5), "roi_exclusive": True,
            "max_time_gap_s": getattr(args, "max_time_gap_s", 30.0),
            "trust_rescue_pair_s": getattr(args, "trust_rescue_pair_s", 4.0),
            "stationary_filter": bool(getattr(args, "stationary_filter", True)),
            "stationary_radius_px": getattr(args, "stationary_radius", 30.0),
            "stationary_min_hits": getattr(args, "stationary_hits", 6),
            "stationary_min_span_s": getattr(args, "stationary_span_s", 300.0),
            "stationary_min_density": getattr(args, "stationary_min_density", 0.6),
            "reuse_detections": bool(reuse),
        },
    }
    (outdir / "scene_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{loc_id}] riders={summary['n_riders']} "
          f"(dominant: sidewalk {summary['riders_dom_sidewalk']}, bike {summary['riders_dom_bike_lane']}, "
          f"road {summary['riders_dom_roadway']}), dir-known {n_dir}, cross {n_cross}, ww {n_ww}, "
          f"stationary-removed {n_stationary}, exif-ts {n_with_ts}/{len(img_paths)}")
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
    ap.add_argument("--max-time-gap-s", type=float, default=30.0,
                    help="EXIF capture-time gap that always breaks association "
                         "(motion-triggered frames can be minutes apart; 0 disables)")
    ap.add_argument("--trust-rescue-pair-s", type=float, default=4.0,
                    help="trust a rescued chain's direction when its EXIF span is "
                         "within this window (the camera's ~2s double-shot of one "
                         "fast rider); 0 restores blanket rescue distrust")
    ap.add_argument("--no-stationary-filter", dest="stationary_filter", action="store_false",
                    help="keep parked-bike clusters (on by default)")
    ap.add_argument("--stationary-radius", type=float, default=30.0,
                    help="cluster radius (px) for the parked-bike filter")
    ap.add_argument("--stationary-hits", type=int, default=6,
                    help="min distinct captures at one spot to call it stationary")
    ap.add_argument("--stationary-span-frames", type=int, default=50,
                    help="min image-number span for a stationary cluster (no-EXIF fallback)")
    ap.add_argument("--stationary-span-s", type=float, default=300.0,
                    help="min EXIF time span (s) for a stationary cluster")
    ap.add_argument("--stationary-min-density", type=float, default=0.6,
                    help="min fraction of the window's captures a cluster must "
                         "appear in; separates parked objects (~1.0) from busy "
                         "chokepoints many riders pass (~0.1-0.3)")
    ap.add_argument("--reuse-detections", action="store_true",
                    help="replay from saved detections_raw.csv instead of "
                         "re-running YOLO (post-processing changes only)")
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
    if "dominant_space" in riders.columns:
        dom = riders["dominant_space"]
        dsw, dbl, drd = (int((dom == k).sum()) for k in ("sidewalk", "bike_lane", "roadway"))
        rows += [
            ("Dominant facility: sidewalk",  f"{dsw}  ({pct(dsw, n)})", ""),
            ("Dominant facility: bike lane", f"{dbl}  ({pct(dbl, n)})", ""),
            ("Dominant facility: roadway",   f"{drd}  ({pct(drd, n)})", ""),
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
        ("  stationary objects removed", -f.get("stationary_object_dets", 0),
         "parked bikes: same spot repeatedly over a long span (stationary_objects.csv)"),
        ("Riders (counting unit)",      f["riders"],                  "nearby-capture association (frame gap + EXIF time gate)"),
        ("Person-only candidates",      f["person_candidates"],       "possible missed riders, flagged for review"),
    ]
    if f.get("captures_with_exif_ts") is not None:
        rows.insert(1, ("  with EXIF timestamp", f["captures_with_exif_ts"],
                        "capture time drives association & stationary spans"))
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


def riders_table(riders, max_rows=60):
    """Riders styled for reading: check marks, colored direction, tabular numbers."""
    import pandas as pd
    d = riders.copy()
    total = len(d)
    if total > max_rows:
        order = {"against_flow": 0, "cross_flow": 1, "along_flow": 2, "unknown": 3}
        d = d.sort_values(by="direction_displacement", key=lambda c: c.map(order)).head(max_rows)
    for c in ("in_sidewalk_any", "in_bike_lane_any", "in_roadway_any"):
        d[c] = d[c].map({True: "\u2713", False: ""})
    if "dominant_space" not in d.columns:
        d["dominant_space"] = ""
    d = d[["rider_id", "n_obs", "img_first", "img_last",
           "in_sidewalk_any", "in_bike_lane_any", "in_roadway_any", "dominant_space",
           "disp_px", "direction_displacement", "cos_to_flow", "wrong_way_displacement"]]
    d.columns = ["rider", "obs", "first capture", "last capture",
                 "sidewalk", "bike lane", "roadway", "dominant",
                 "disp (px)", "direction", "cos", "wrong-way"]
    def dir_color(v):
        return f"color: {DIRECTION_COLORS.get(v, _INK)}; font-weight: 600;"
    sty = (d.style.hide(axis="index")
           .set_caption(f"Riders — one row per counted rider"
                        + (f" (top {len(d)} of {total}, direction-known first; full list in riders.csv)"
                           if total > len(d) else f" (n={total})"))
           .format({"disp (px)": "{:.0f}", "cos": lambda v: "" if pd.isna(v) else f"{v:+.2f}",
                    "wrong-way": lambda v: "" if pd.isna(v) else ("YES" if v else "no")}, na_rep="")
           .map(dir_color, subset=["direction"])
           .map(lambda v: f"color: {FACILITY_COLORS.get(v, _INK2)}; font-weight: 600;",
                subset=["dominant"])
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
    total = len(q)
    q = q.head(30)
    q = q[["img", "score", "space_label", "in_study_roi"]]
    q.columns = ["capture", "confidence", "ROI label", "in study ROI"]
    sty = (q.style.hide(axis="index")
           .set_caption(f"Person-only candidates — top {len(q)} of {total} by (in-ROI, confidence); full list in person_candidates.csv")
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
    if "assoc_rescue" not in d.columns:
        d["assoc_rescue"] = False
    d["assoc"] = d["assoc_rescue"].map({True: "rescue", False: "gated"})
    d = d[["rider_id", "img_first", "img_last", "disp_px", "cos_to_flow",
           "direction_displacement", "wrong_way_displacement", "assoc", "check image"]]
    d.columns = ["rider", "first frame", "last frame", "disp (px)", "cos",
                 "direction", "wrong-way", "assoc", "check image"]
    def dir_color(v):
        return f"color: {DIRECTION_COLORS.get(v, _INK)}; font-weight: 700;"
    sty = (d.style.hide(axis="index")
           .set_caption(f"Direction verdicts — verify each against direction_check/ (n={len(d)})")
           .format({"disp (px)": "{:.0f}", "cos": "{:+.2f}",
                    "wrong-way": lambda v: "YES" if v is True or v == True else "no"}, na_rep="")
           .map(dir_color, subset=["direction"])
           .map(lambda v: "background-color: #fdecea; font-weight: 700;" if v == "YES" else "",
                subset=["wrong-way"])
           .map(lambda v: f"color: {_INK2}; font-style: italic;" if v == "rescue" else "",
                subset=["assoc"])
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


def generate_review_html(outdir, loc_id):
    """One-page manual review tool. Open in a browser from the output folder,
    click one big button per image, Export downloads review_<loc>.csv."""
    import re as _re
    import pandas as pd
    outdir = Path(outdir)
    riders = pd.read_csv(outdir / "riders.csv") if (outdir / "riders.csv").exists() else pd.DataFrame()
    crops = sorted((outdir / "crops").glob("*.jpg")) if (outdir / "crops").exists() else []
    pcrops = sorted((outdir / "crops_person").glob("*.jpg")) if (outdir / "crops_person").exists() else []

    rider_first_crop = {}
    for f in crops:
        rid = int(f.name.split("_")[0].replace("rider", ""))
        rider_first_crop.setdefault(rid, f"crops/{f.name}")

    # counted-rider frames, for near-rider warnings on candidates
    rider_frames = {}
    dr_path = outdir / "detections_riders.csv"
    if dr_path.exists():
        _dr = pd.read_csv(dr_path)
        if "img_num" in _dr.columns and "rider_id" in _dr.columns:
            for _, _r in _dr.iterrows():
                rider_frames.setdefault(int(_r.img_num), set()).add(int(_r.rider_id))

    def near_rider(frame_num, win=3):
        hits = set()
        for f, rids in rider_frames.items():
            if abs(f - frame_num) <= win:
                hits |= rids
        return sorted(hits)

    # collapse near-duplicate person crops (same frame, x within 80px)
    kept, last = [], {}
    for f in pcrops:
        m = _re.match(r"(IM_\d+)_p(\d+)", f.stem)
        if not m:
            kept.append(f); continue
        frame, x = m.group(1), int(m.group(2))
        if frame in last and abs(x - last[frame]) < 80:
            continue
        last[frame] = x
        kept.append(f)

    def item(img_rel, iid, kind, warn=""):
        w = ('<div class="warn">' + warn + '</div>') if warn else ""
        return ('<div class="item" data-id="' + iid + '" data-kind="' + kind + '">'
                '<img src="' + img_rel + '" loading="lazy"><div class="meta">' + iid + '</div>'
                + w + '<div class="btns"></div></div>')

    rider_items = [item(rider_first_crop[int(r.rider_id)], "R" + str(int(r.rider_id)).zfill(4), "rider")
                   for _, r in riders.iterrows() if int(r.rider_id) in rider_first_crop]
    cand_items = []
    for f in kept:
        m = _re.match(r"IM_(\d+)_p\d+", f.stem)
        warn = ""
        if m:
            hits = near_rider(int(m.group(1)))
            if hits:
                warn = "&#9888; ±3帧内已有计数rider R" + ", R".join(str(h) for h in hits) + " — 可能是同一人"
        cand_items.append(item("crops_person/" + f.name, f.stem, "candidate", warn))

    html_parts = ["""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Review LOCID</title>
<style>
 body{font-family:system-ui,sans-serif;margin:14px;background:#fcfcfb;color:#0b0b0b}
 h2{border-bottom:2px solid #0b0b0b;padding-bottom:4px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
 .item{background:#fff;border:1px solid #e5e5e2;border-radius:10px;padding:10px}
 .item img{width:100%;max-height:260px;object-fit:contain;background:#eee;border-radius:6px}
 .meta{font-size:12px;color:#52514e;margin:6px 0 4px}
 .warn{font-size:12px;color:#b3261e;background:#fdecea;border-radius:6px;padding:4px 6px;margin:4px 0}
 .btns button{margin:3px;padding:10px 12px;font-size:15px;border:1px solid #bbb;border-radius:8px;background:#f6f6f4;cursor:pointer}
 .btns button:hover{background:#e8f0fb}
 .btns button.sel{background:#2a78d6;color:#fff;border-color:#2a78d6}
 #bar{position:sticky;top:0;background:#fcfcfb;padding:10px 0;border-bottom:1px solid #e5e5e2;z-index:9}
 #bar button{padding:10px 18px;font-weight:700;font-size:15px;border-radius:8px;border:2px solid #0b0b0b;background:#fff;cursor:pointer}
 .done{outline:3px solid #1baf7a}
</style></head><body>
<div id="bar"><b>Review LOCID</b>
 <button onclick="exportCsv()">Export CSV</button> <span id="prog"></span>
 <span style="color:#52514e;font-size:13px">每张图点一个按钮;进度自动保存;全部标完点 Export</span></div>
<h2>A. riders (""" + str(len(rider_items)) + """ 个) — 相对车流方向: 顺流还是逆流?</h2>
<div class="grid">""", "\n".join(rider_items) or "<i>无裁剪图 (需 save_crops=True)</i>", """</div>
<h2>B. person 候选 (""" + str(len(cand_items)) + """ 个) — 是漏检的骑行者吗? 顺流还是逆流?</h2>
<div class="grid">""", "\n".join(cand_items) or "<i>无</i>", """</div>
<script>
var NL = String.fromCharCode(10);
var OPTS = {rider:["顺流","逆流","横穿","不是骑行者","看不清"],
            candidate:["漏检骑行者:顺流","漏检骑行者:逆流","已计数(同一人)","不是骑行者","看不清"]};
var KEY = "review_LOCID";
var store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { store = {}; }
var items = document.querySelectorAll(".item");
for (var i = 0; i < items.length; i++) (function(it){
  var id = it.getAttribute("data-id"), kind = it.getAttribute("data-kind");
  var bx = it.querySelector(".btns"), opts = OPTS[kind];
  for (var j = 0; j < opts.length; j++) (function(o){
    var b = document.createElement("button");
    b.textContent = o;
    if (store[id] === o) { b.className = "sel"; it.classList.add("done"); }
    b.onclick = function(){
      store[id] = o;
      try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {}
      var bs = bx.querySelectorAll("button");
      for (var k = 0; k < bs.length; k++) bs[k].className = "";
      b.className = "sel"; it.classList.add("done"); prog();
    };
    bx.appendChild(b);
  })(opts[j]);
})(items[i]);
function prog(){
  var d = 0; for (var k in store) d++;
  document.getElementById("prog").textContent = " " + d + "/" + items.length;
}
prog();
function exportCsv(){
  var rows = ["item_id,kind,label"];
  for (var i = 0; i < items.length; i++) {
    var it = items[i], id = it.getAttribute("data-id");
    rows.push(id + "," + it.getAttribute("data-kind") + "," + (store[id] || ""));
  }
  var a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.join(NL)], {type: "text/csv"}));
  a.download = "review_LOCID.csv"; a.click();
}
</script></body></html>"""]
    html = "".join(html_parts).replace("LOCID", loc_id)
    out = outdir / f"review_{loc_id}.html"
    out.write_text(html, encoding="utf-8")
    return out, len(rider_items), len(cand_items)


# ---------------------------------------------------------------------------
# Cross-location aggregation, manual comparison, review-CSV metrics
# ---------------------------------------------------------------------------
def aggregate_locations(out_root, exclude=()):
    """One row per completed location under out_root."""
    import json as _json
    import pandas as pd
    rows = []
    for d in sorted(Path(out_root).glob("loc_*")):
        if not d.is_dir() or d.name in exclude:
            continue
        sj = d / "scene_summary.json"
        rc = d / "riders.csv"
        if not (sj.exists() and rc.exists()):
            continue
        summ = _json.loads(sj.read_text())
        r = pd.read_csv(rc)
        dd = r["direction_displacement"]
        along = int((dd == "along_flow").sum())
        against = int((dd == "against_flow").sum())
        dk = along + against
        lo, hi = _wilson(against, dk) if dk else (0, 0)
        rows.append({
            "location": d.name,
            "captures": summ.get("funnel", {}).get("captures"),
            "riders": len(r),
            "sidewalk": int(r["in_sidewalk_any"].sum()),
            "bike_lane": int(r["in_bike_lane_any"].sum()),
            "roadway": int(r["in_roadway_any"].sum()),
            "dom_sidewalk": int((r["dominant_space"] == "sidewalk").sum()) if "dominant_space" in r.columns else None,
            "dom_bike_lane": int((r["dominant_space"] == "bike_lane").sum()) if "dominant_space" in r.columns else None,
            "dom_roadway": int((r["dominant_space"] == "roadway").sum()) if "dominant_space" in r.columns else None,
            "dir_known": dk,
            "along": along,
            "against": against,
            "cross": int((dd == "cross_flow").sum()),
            "ww_rate": against / dk if dk else None,
            "ww_lo": lo if dk else None,
            "ww_hi": hi if dk else None,
            "coverage": dk / len(r) if len(r) else 0,
            "person_cands": summ.get("funnel", {}).get("person_candidates"),
        })
    return pd.DataFrame(rows)


def all_locations_table(df):
    import pandas as pd
    d = df.copy()
    d["WW rate"] = d.apply(lambda r: "" if not r.dir_known else
                           f"{r.ww_rate:.0%} ({r.against}/{r.dir_known}, CI {r.ww_lo:.0%}-{r.ww_hi:.0%})", axis=1)
    d["coverage"] = d["coverage"].map(lambda v: f"{v:.0%}")
    cols = ["location", "captures", "riders", "sidewalk", "bike_lane", "roadway",
            "dom_sidewalk", "dom_bike_lane", "dom_roadway",
            "dir_known", "coverage", "along", "against", "cross", "WW rate", "person_cands"]
    d = d[[c for c in cols if c in d.columns]]
    sty = (d.style.hide(axis="index")
           .set_caption(f"All locations — pipeline results ({len(d)} locations)")
           .map(lambda v: "background-color: #fdecea; font-weight:600;" if isinstance(v, str) and "CI" in v else "",
                subset=["WW rate"])
           .set_properties(subset=[c for c in d.columns if c not in ("location", "WW rate")],
                           **{"text-align": "right", "font-variant-numeric": "tabular-nums"})
           .set_table_styles([
               {"selector": "caption", "props": f"caption-side: top; font-weight:600; color:{_INK}; padding:6px 0;"},
               {"selector": "th", "props": f"text-align:left; color:{_INK2}; border-bottom:1.5px solid {_INK}; padding:4px 8px;"},
               {"selector": "td", "props": f"padding:3px 8px; border-bottom:0.5px solid {_GRID};"},
           ]))
    return sty


def manual_comparison_table(df, manual):
    """manual: {location: (along, against)}. Deviation-ranked comparison."""
    import pandas as pd
    rows = []
    for _, r in df.iterrows():
        m = manual.get(r.location)
        if not m:
            continue
        ma, mg = m
        md = ma + mg
        mww = mg / md if md else None
        rows.append({
            "location": r.location,
            "riders (pipe)": r.riders, "riders (manual)": md,
            "recall": r.riders / md if md else None,
            "WW pipe": r.ww_rate, "WW manual": mww,
            "ΔWW (pp)": (r.ww_rate - mww) * 100 if (r.ww_rate is not None and mww is not None) else None,
            "dir coverage": r.coverage,
        })
    d = pd.DataFrame(rows)
    if len(d):
        d = d.sort_values("ΔWW (pp)", key=lambda c: c.abs(), ascending=False)
    sty = (d.style.hide(axis="index")
           .set_caption("Pipeline vs manual benchmark — sorted by |ΔWW| (largest deviation first)")
           .format({"recall": "{:.0%}", "WW pipe": "{:.1%}", "WW manual": "{:.1%}",
                    "ΔWW (pp)": "{:+.1f}", "dir coverage": "{:.0%}"}, na_rep="-")
           .map(lambda v: "background-color:#fdecea; font-weight:700;" if isinstance(v, float) and abs(v) > 8 else "",
                subset=["ΔWW (pp)"])
           .set_table_styles([
               {"selector": "caption", "props": f"caption-side: top; font-weight:600; color:{_INK}; padding:6px 0;"},
               {"selector": "th", "props": f"text-align:left; color:{_INK2}; border-bottom:1.5px solid {_INK}; padding:4px 8px;"},
               {"selector": "td", "props": f"padding:3px 8px; border-bottom:0.5px solid {_GRID};"},
           ]))
    return sty


def facility_comparison_table(df, manual_df):
    """Dominant-facility counts vs the manual benchmark, like-for-like.
    manual_df = data/manual_counts_new.csv (fwd_bl/ww_bl etc.); manual facility
    total = fwd + ww for that facility. Pipeline side uses dominant_space (the
    facility the rider mainly rode on), the same definition the human count
    used — the any-involvement columns are intentionally NOT compared here."""
    import pandas as pd
    rows = []
    need = ["fwd_bl", "fwd_sw", "fwd_rd", "ww_bl", "ww_sw", "ww_rd"]
    for _, m in manual_df.dropna(subset=need).iterrows():
        loc = f"loc_{m['loc']}"
        hit = df[df.location == loc]
        if not len(hit):
            continue
        r = hit.iloc[0]
        rows.append({
            "location": loc,
            "BL pipe": r.get("dom_bike_lane"),
            "BL manual": int(m.fwd_bl) + int(m.ww_bl),
            "SW pipe": r.get("dom_sidewalk"),
            "SW manual": int(m.fwd_sw) + int(m.ww_sw),
            "RD pipe": r.get("dom_roadway"),
            "RD manual": int(m.fwd_rd) + int(m.ww_rd),
        })
    d = pd.DataFrame(rows)
    if not len(d):
        return None
    tot = {"location": "TOTAL"}
    for c in d.columns[1:]:
        tot[c] = int(pd.to_numeric(d[c], errors="coerce").fillna(0).sum())
    d = pd.concat([d, pd.DataFrame([tot])], ignore_index=True)
    sty = (d.style.hide(axis="index")
           .set_caption("Facility split, pipeline dominant_space vs manual (same definition: "
                        "facility the rider mainly used)")
           .format(na_rep="-", precision=0)
           .map(lambda v: "font-weight:700;", subset=pd.IndexSlice[d.index[d.location == "TOTAL"], :])
           .set_properties(subset=[c for c in d.columns if c != "location"],
                           **{"text-align": "right", "font-variant-numeric": "tabular-nums"})
           .set_table_styles([
               {"selector": "caption", "props": f"caption-side: top; font-weight:600; color:{_INK}; padding:6px 0;"},
               {"selector": "th", "props": f"text-align:left; color:{_INK2}; border-bottom:1.5px solid {_INK}; padding:4px 8px;"},
               {"selector": "td", "props": f"padding:3px 8px; border-bottom:0.5px solid {_GRID};"},
           ]))
    return sty


def pair_diagnostic(outdir, img_dir, loc_id="", pair_window_s=4.0):
    """Why is a rider's direction unknown? Decompose the coverage gap.

    The cameras shoot two images ~2 s apart per trigger (per Deborah), so in
    principle almost every event should have a displacement pair. This audit
    classifies every direction-unknown rider into:
      no_pair_on_disk      no second capture within pair_window_s exists at all
      pair_missed          a paired capture exists but the rider was not
                           detected in it (blur / occlusion / left the frame)
      too_little_motion    seen in >=2 frames but moved < min_move_px
      rescue_distrusted    associated by burst rescue; direction withheld
    Reads EXIF times of all captures once and caches them to
    outdir/capture_index.csv so re-runs are instant.
    """
    outdir = Path(outdir)
    riders = pd.read_csv(outdir / "riders.csv")
    idx_csv = outdir / "capture_index.csv"
    if idx_csv.exists():
        cap = pd.read_csv(idx_csv)
    else:
        rows = [{"img": p.name, "img_num": imgnum(p.name), "ts": read_exif_ts(p)}
                for p in collect_images(Path(img_dir))]
        cap = pd.DataFrame(rows)
        cap.to_csv(idx_csv, index=False)
    ts_by_num = {int(r.img_num): r.ts for _, r in cap.iterrows() if pd.notna(r.ts)}

    def has_pair(img_num):
        t = ts_by_num.get(int(img_num))
        if t is None:
            return None   # no EXIF: cannot audit this one
        return any(abs(ts_by_num[n] - t) <= pair_window_s
                   for n in (img_num - 1, img_num + 1) if n in ts_by_num)

    unknown = riders[~riders["direction_displacement"].isin(
        ["along_flow", "against_flow", "cross_flow"])]
    counts = {"no_pair_on_disk": 0, "pair_missed": 0, "too_little_motion": 0,
              "rescue_distrusted": 0, "no_exif": 0}
    for _, r in unknown.iterrows():
        if r.get("assoc_rescue") is True or r.get("assoc_rescue") == True:
            counts["rescue_distrusted"] += 1
        elif int(r.n_obs) >= 2:
            counts["too_little_motion"] += 1
        else:
            p = has_pair(int(r.img_num_first))
            if p is None:
                counts["no_exif"] += 1
            elif p:
                counts["pair_missed"] += 1
            else:
                counts["no_pair_on_disk"] += 1
    n_unknown = len(unknown)
    res = {"location": loc_id or outdir.name, "riders": len(riders),
           "direction_known": len(riders) - n_unknown, "unknown": n_unknown}
    res.update(counts)
    if n_unknown:
        res["pct_pair_missed"] = counts["pair_missed"] / n_unknown
        res["pct_no_pair"] = counts["no_pair_on_disk"] / n_unknown
    return res


def analyze_review_csv(outdir, loc_id):
    """Metrics from a labeled review_<loc>.csv exported by the review page."""
    import pandas as pd
    f = Path(outdir) / f"review_{loc_id}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d["label"] = d["label"].fillna("")
    A = d[d.kind == "rider"]
    B = d[d.kind == "candidate"]
    a_lab = A[A.label != ""]
    fp = int((a_lab.label == "不是骑行者").sum())
    unclear = int((a_lab.label == "看不清").sum())
    tp = len(a_lab) - fp - unclear
    along = int((a_lab.label == "顺流").sum())
    against = int((a_lab.label == "逆流").sum())
    b_lab = B[B.label != ""]
    missed_a = int((b_lab.label == "漏检骑行者:顺流").sum())
    missed_g = int((b_lab.label == "漏检骑行者:逆流").sum())
    dup = int((b_lab.label == "已计数(同一人)").sum())
    res = {
        "location": loc_id,
        "riders_labeled": len(a_lab),
        "precision": tp / max(len(a_lab) - unclear, 1),
        "false_positives": fp,
        "manual_along": along, "manual_against": against,
        "manual_ww": against / max(along + against, 1),
        "missed_riders": missed_a + missed_g,
        "missed_along": missed_a, "missed_against": missed_g,
        "duplicates_flagged": dup,
        "corrected_riders": tp + missed_a + missed_g,
        "corrected_ww": (against + missed_g) / max(along + against + missed_a + missed_g, 1),
    }
    return res

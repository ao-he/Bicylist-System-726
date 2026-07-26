# src/inference/common.py
def frame_space_label(bb: BBox, roi: ROIMaskEngine, thr: FrameLabelThresholds) -> str:
    bc = roi.bbox_bottom_center(bb)

    def has_roi(name: str) -> bool:
        return bool(roi.cfg.rois.get(name)) and len(roi.cfg.rois[name]) > 0

    # --- 0) ignore_zone: if exists and point-in => unknown (or "ignore") ---
    if has_roi("ignore_zone"):
        if roi.point_in_roi(bc, "ignore_zone"):
            return "unknown"

    # --- 1) Crosswalk: bottom-edge 3-point rule (robust) ---
    if has_roi("crosswalk"):
        x1, y1, x2, y2 = bb
        xs = [x1, (x1 + x2) / 2.0, x2]
        pts = [(float(x), float(y2)) for x in xs]
        if any(roi.point_in_roi(p, "crosswalk") for p in pts):
            return "crosswalk"

    # --- 2) Bike lane ---
    if has_roi("bike_lane"):
        ov = roi.overlap_ratio(bb, "bike_lane", shrink=thr.shrink_frac)
        if ov > thr.T_BIKE:
            if (not thr.use_bottom_center_gate) or roi.point_in_roi(bc, "bike_lane"):
                return "bike_lane"

    # --- 3) Sidewalk ---
    if has_roi("sidewalk"):
        ov = roi.overlap_ratio(bb, "sidewalk", shrink=thr.shrink_frac)
        if ov > thr.T_SIDE:
            if (not thr.use_bottom_center_gate) or roi.point_in_roi(bc, "sidewalk"):
                return "sidewalk"

    # --- 4) Roadway ---
    if has_roi("roadway"):
        ov = roi.overlap_ratio(bb, "roadway", shrink=thr.shrink_frac)
        if ov > thr.T_ROAD:
            return "roadway"

    return "unknown"

# src/scene/roi.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import cv2


# -----------------------------
# Types
# -----------------------------
BBox = Tuple[float, float, float, float]     # (x1, y1, x2, y2)
Point = Tuple[float, float]                 # (x, y) in image coords
Polygon = List[Point]                       # list of points


# -----------------------------
# Config model
# -----------------------------
@dataclass(frozen=True)
class ROIConfig:
    """
    Parsed config for one location instance.

    Expected JSON schema (from your ROI editor):
    {
      "location_id": "Loc_1",
      "image_size": {"w": 1920, "h": 1080},
      "rois": {
        "roadway":   [ [ [x,y], ... ], ... ],
        "bike_lane": [ ... ],
        "sidewalk":  [ ... ],
        "crosswalk": [ ... ],
        "ignore_zone":[ ... ]
      },
      "flow": { "type":"arrow", "vector": {"x1":..,"y1":..,"x2":..,"y2":..} } | null
    }
    """
    location_id: str
    w: int
    h: int
    rois: Dict[str, List[Polygon]]
    flow: Optional[dict] = None


def load_roi_config(json_path: Union[str, Path]) -> ROIConfig:
    """
    Load a location ROI config exported by the HTML ROI editor.
    """
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))

    location_id = str(data.get("location_id", p.stem))
    w = int(data["image_size"]["w"])
    h = int(data["image_size"]["h"])

    rois = data.get("rois", {})
    # Ensure keys exist even if empty
    for k in ["roadway", "bike_lane", "sidewalk", "crosswalk", "ignore_zone"]:
        rois.setdefault(k, [])

    return ROIConfig(location_id=location_id, w=w, h=h, rois=rois, flow=data.get("flow"))


# -----------------------------
# ROI Engine
# -----------------------------
class ROIMaskEngine:
    """
    Converts ROI polygons into binary masks and provides fast queries:
      - point_in_roi(point, roi_type)
      - overlap_ratio(bbox, roi_type) = area(bbox ∩ ROI) / area(bbox)

    Notes:
      - This file is intentionally "dumb geometry". No behavior rules here.
      - Keep stable: everything else depends on this.
    """

    def __init__(self, cfg: ROIConfig):
        self.cfg = cfg
        self.masks: Dict[str, np.ndarray] = {}
        self.roi_area: Dict[str, int] = {}
        self._build_masks()

    # ---------- build ----------
    def _build_masks(self) -> None:
        h, w = self.cfg.h, self.cfg.w
        for roi_type, polygons in self.cfg.rois.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            for poly in polygons:
                if len(poly) < 3:
                    continue
                pts = np.array([[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))  # OpenCV expects (N,1,2)
                cv2.fillPoly(mask, [pts], 1)
            self.masks[roi_type] = mask
            self.roi_area[roi_type] = int(mask.sum())

    # ---------- helpers ----------
    @staticmethod
    def bbox_bottom_center(bbox: BBox) -> Point:
        x1, y1, x2, y2 = bbox
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return ((x1 + x2) / 2.0, y2)

    @staticmethod
    def shrink_bbox(bbox: BBox, frac: float = 0.08) -> BBox:
        """
        Shrink bbox by a fraction on each side to reduce boundary noise.
        frac=0.08 means ~8% shrink on each side.
        """
        x1, y1, x2, y2 = bbox
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        w = x2 - x1
        h = y2 - y1
        dx = w * frac
        dy = h * frac
        return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)

    @staticmethod
    def _clip_bbox(bbox: BBox, w: int, h: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        x1i = int(max(0, min(w - 1, np.floor(x1))))
        y1i = int(max(0, min(h - 1, np.floor(y1))))
        x2i = int(max(0, min(w, np.ceil(x2))))   # allow == w
        y2i = int(max(0, min(h, np.ceil(y2))))   # allow == h

        if x2i <= x1i:
            x2i = min(w, x1i + 1)
        if y2i <= y1i:
            y2i = min(h, y1i + 1)
        return x1i, y1i, x2i, y2i

    # ---------- queries ----------
    def point_in_roi(self, pt: Point, roi_type: str) -> bool:
        """
        Return True if pt lies inside roi_type.
        """
        mask = self.masks.get(roi_type)
        if mask is None:
            return False
        x, y = pt
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or yi < 0 or xi >= self.cfg.w or yi >= self.cfg.h:
            return False
        return bool(mask[yi, xi] == 1)

    def overlap_ratio(self, bbox: BBox, roi_type: str, shrink: float = 0.0) -> float:
        """
        area(bbox ∩ ROI) / area(bbox)
        """
        mask = self.masks.get(roi_type)
        if mask is None:
            return 0.0

        bb = self.shrink_bbox(bbox, shrink) if shrink and shrink > 0 else bbox
        x1, y1, x2, y2 = self._clip_bbox(bb, self.cfg.w, self.cfg.h)

        box_area = float((x2 - x1) * (y2 - y1))
        if box_area <= 0:
            return 0.0

        inter = float(mask[y1:y2, x1:x2].sum())
        return inter / box_area

    # ---------- flow ----------
    def flow_vector(self) -> Optional[Tuple[float, float]]:
        """
        Returns a normalized (fx, fy) direction vector from flow arrow, or None.
        """
        if not self.cfg.flow:
            return None
        vec = self.cfg.flow.get("vector")
        if not vec:
            return None

        fx = float(vec["x2"]) - float(vec["x1"])
        fy = float(vec["y2"]) - float(vec["y1"])
        norm = (fx * fx + fy * fy) ** 0.5
        if norm < 1e-6:
            return None
        return (fx / norm, fy / norm)

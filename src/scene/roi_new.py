# src/scene/roi_new.py
"""
NEW-pipeline ROI engine. The original src/scene/roi.py is left untouched;
this subclass adds overlap resolution on top of it.

Overlapping polygons are resolved by priority (highest first):
    crosswalk > bike_lane > sidewalk > roadway
Each lower-priority mask has all higher-priority regions subtracted, so the
annotator may freely draw e.g. the roadway across the whole street with the
bike lane / sidewalk layered on top.
"""
from __future__ import annotations

import numpy as np

from src.scene.roi import ROIConfig, ROIMaskEngine, load_roi_config  # noqa: F401


class ExclusiveROIMaskEngine(ROIMaskEngine):
    OVERLAP_PRIORITY = ("crosswalk", "bike_lane", "sidewalk", "roadway")

    def __init__(self, cfg: ROIConfig):
        super().__init__(cfg)
        claimed = None
        for roi_type in self.OVERLAP_PRIORITY:
            mask = self.masks.get(roi_type)
            if mask is None:
                continue
            if claimed is not None:
                mask = (mask & ~claimed).astype(np.uint8)
                self.masks[roi_type] = mask
                self.roi_area[roi_type] = int(mask.sum())
            claimed = mask if claimed is None else (claimed | mask)

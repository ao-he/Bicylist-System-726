"""Unit tests for the NEW rider-counting pipeline.
Run:  python tests/test_pipeline_new.py   (or pytest tests/)
"""
import sys, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("run_rider_count_new", ROOT / "scripts" / "run_rider_count_new.py")
rrc = importlib.util.module_from_spec(spec)
sys.modules["run_rider_count_new"] = rrc
spec.loader.exec_module(rrc)

import pandas as pd

PASS = []

def check(name, cond):
    PASS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


def test_containment_dedup():
    det = pd.DataFrame([
        dict(img="a", img_num=1, x1=100, y1=100, x2=200, y2=400, score=0.9),
        dict(img="a", img_num=1, x1=110, y1=250, x2=190, y2=395, score=0.5),
        dict(img="a", img_num=1, x1=600, y1=100, x2=700, y2=400, score=0.8),
    ])
    out = rrc.dedup_contained_per_frame(det)
    check("containment dedup keeps 2 of 3", len(out) == 2)


def test_association_and_rescue():
    h_red = [1.0] + [0.0] * 63
    h_grn = [0.0] * 32 + [1.0] + [0.0] * 31
    det = pd.DataFrame([
        dict(img="z", img_num=900, x1=1800, y1=100, x2=1900, y2=200, score=.9, space_label="roadway", app=h_red),
        dict(img="a", img_num=1, x1=200, y1=300, x2=310, y2=460, score=.9, space_label="bike_lane", app=h_red),
        dict(img="b", img_num=2, x1=1200, y1=330, x2=1260, y2=420, score=.9, space_label="bike_lane", app=h_grn),
        dict(img="c", img_num=10, x1=200, y1=300, x2=310, y2=460, score=.9, space_label="bike_lane", app=h_red),
        dict(img="d", img_num=11, x1=1200, y1=330, x2=1260, y2=420, score=.9, space_label="bike_lane", app=h_red),
    ])
    out = rrc.associate_riders(det, rrc.AssocParams())
    check("appearance gate blocks dissimilar rescue", out[out.img_num.isin([1, 2])].rider_id.nunique() == 2)
    check("appearance gate allows similar rescue", out[out.img_num.isin([10, 11])].rider_id.nunique() == 1)
    check("rescued detection flagged", bool(out[out.img_num == 11].assoc_rescue.iloc[0]))


def test_direction_rules():
    flow = (1.0, 0.0)
    det = pd.DataFrame([  # along mover, gated
        dict(img="a", img_num=1, x1=100, y1=500, x2=160, y2=600, score=.9, space_label="bike_lane", assoc_rescue=False),
        dict(img="b", img_num=2, x1=300, y1=500, x2=360, y2=600, score=.9, space_label="bike_lane", assoc_rescue=False),
    ]).assign(rider_id=0)
    r = rrc.summarize_riders(det, flow=flow, params=rrc.AssocParams())
    check("along flow detected", r.direction_displacement.iloc[0] == "along_flow")

    det2 = det.copy(); det2.loc[1, ["x1", "x2"]] = [104, 164]  # 4px jitter
    r2 = rrc.summarize_riders(det2, flow=flow, params=rrc.AssocParams())
    check("stationary -> unknown (25px gate)", r2.direction_displacement.iloc[0] == "unknown")

    det3 = det.copy(); det3.loc[1, ["x1", "x2", "y1", "y2"]] = [100, 160, 800, 900]  # perpendicular
    r3 = rrc.summarize_riders(det3, flow=flow, params=rrc.AssocParams())
    check("perpendicular -> cross_flow", r3.direction_displacement.iloc[0] == "cross_flow")

    det4 = det.copy(); det4["assoc_rescue"] = [False, True]  # rescued pair
    r4 = rrc.summarize_riders(det4, flow=flow, params=rrc.AssocParams())
    check("rescued pair -> no direction", r4.direction_displacement.iloc[0] == "unknown")


def test_roi_priority():
    import json, tempfile
    cfg_json = {"location_id": "syn", "image_size": {"w": 1000, "h": 1000},
                "rois": {"roadway": [[[0, 0], [1000, 0], [1000, 1000], [0, 1000]]],
                         "bike_lane": [[[0, 500], [1000, 500], [1000, 600], [0, 600]]],
                         "sidewalk": [], "crosswalk": [], "ignore_zone": []},
                "flow": None}
    f = Path(tempfile.mkdtemp()) / "syn.json"
    f.write_text(json.dumps(cfg_json))
    from src.scene.roi_new import load_roi_config, ExclusiveROIMaskEngine
    roi = ExclusiveROIMaskEngine(load_roi_config(f))
    from src.inference.common import frame_space_label, FrameLabelThresholds
    thr = FrameLabelThresholds(bottom_edge_npts=5, vote_min_pts=2)
    check("bike lane wins over full-cover roadway",
          frame_space_label((200, 450, 300, 550), roi, thr) == "bike_lane")


if __name__ == "__main__":
    test_containment_dedup()
    test_association_and_rescue()
    test_direction_rules()
    test_roi_priority()
    n = sum(PASS)
    print(f"\n{n}/{len(PASS)} tests passed")
    sys.exit(0 if n == len(PASS) else 1)

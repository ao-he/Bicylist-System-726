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

    det4 = det.copy(); det4["assoc_rescue"] = [False, True]  # rescued pair, no EXIF
    r4 = rrc.summarize_riders(det4, flow=flow, params=rrc.AssocParams())
    check("rescued pair -> no direction", r4.direction_displacement.iloc[0] == "unknown")

    # rescued pair INSIDE the camera's 2s double-shot window -> trusted
    det5 = det4.copy(); det5["ts"] = [1000.0, 1002.0]
    r5 = rrc.summarize_riders(det5, flow=flow, params=rrc.AssocParams())
    check("2s rescue pair -> direction trusted", r5.direction_displacement.iloc[0] == "along_flow")
    check("2s rescue pair tagged pair_rescue", r5.direction_source.iloc[0] == "pair_rescue")

    # rescued pair 20s apart (queue risk) -> still distrusted
    det6 = det4.copy(); det6["ts"] = [1000.0, 1020.0]
    r6 = rrc.summarize_riders(det6, flow=flow, params=rrc.AssocParams())
    check("20s rescue pair -> still unknown", r6.direction_displacement.iloc[0] == "unknown")

    # opt-out restores blanket distrust
    r7 = rrc.summarize_riders(det5, flow=flow, params=rrc.AssocParams(trust_rescue_pair_s=0))
    check("trust_rescue_pair_s=0 disables", r7.direction_displacement.iloc[0] == "unknown")


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


def test_time_gap_gate():
    # consecutive image numbers, but EXIF says 10 minutes apart:
    # two different people, must NOT merge into one rider
    det = pd.DataFrame([
        dict(img="a", img_num=1, x1=200, y1=300, x2=310, y2=460, score=.9,
             space_label="bike_lane", ts=1000.0),
        dict(img="b", img_num=2, x1=230, y1=300, x2=340, y2=460, score=.9,
             space_label="bike_lane", ts=1600.0),
    ])
    out = rrc.associate_riders(det, rrc.AssocParams(max_time_gap_s=30.0))
    check("EXIF gap breaks association", out.rider_id.nunique() == 2)

    det2 = det.copy(); det2["ts"] = [1000.0, 1006.0]  # 6 s apart: same rider
    out2 = rrc.associate_riders(det2, rrc.AssocParams(max_time_gap_s=30.0))
    check("close EXIF times keep association", out2.rider_id.nunique() == 1)

    out3 = rrc.associate_riders(det, rrc.AssocParams(max_time_gap_s=0))
    check("time gate disabled with 0", out3.rider_id.nunique() == 1)


def test_stationary_filter():
    # parked bike: same spot in 8 captures spread over 40 minutes
    rows = [dict(img=f"p{i}", img_num=n, x1=500, y1=300, x2=560, y2=400,
                 score=.9, ts=1000.0 + i * 300.0)
            for i, n in enumerate([1, 30, 61, 95, 130, 168, 200, 240])]
    # real rider passing nearby (3 captures, 10 s, never on the parked spot)
    rows += [dict(img=f"r{i}", img_num=300 + i, x1=100 + 250 * i, y1=310,
                  x2=160 + 250 * i, y2=410, score=.9, ts=9000.0 + 5.0 * i)
             for i in range(3)]
    det = pd.DataFrame(rows)
    flags = rrc.flag_stationary_detections(det)
    check("parked bike flagged", flags[:8].all())
    check("moving rider spared", not flags[8:].any())

    # waiting cyclist: same spot but only 60 s span -> NOT stationary
    det2 = pd.DataFrame([dict(img=f"w{i}", img_num=1 + i, x1=500, y1=300,
                              x2=560, y2=400, score=.9, ts=1000.0 + 10.0 * i)
                         for i in range(6)])
    check("waiting cyclist (60s) spared", not rrc.flag_stationary_detections(det2).any())

    # busy chokepoint (the loc_17 failure): 10 DIFFERENT riders hit the same
    # spot of a narrow bike lane over an hour, but 90 other captures in the
    # window show riders elsewhere -> density is low, cluster must survive
    rows3 = [dict(img=f"c{i}", img_num=1 + 10 * i, x1=500, y1=300, x2=560,
                  y2=400, score=.9, ts=1000.0 + 360.0 * i) for i in range(10)]
    rows3 += [dict(img=f"o{i}", img_num=2 + i, x1=900 + 7 * i, y1=310,
                   x2=960 + 7 * i, y2=410, score=.9, ts=1005.0 + 36.0 * i)
              for i in range(90)]
    det3 = pd.DataFrame(rows3)
    flags3 = rrc.flag_stationary_detections(det3)
    check("busy chokepoint NOT flagged (density gate)", not flags3[:10].any())


def test_dominant_space():
    det = pd.DataFrame([  # 2 sidewalk frames, 1 roadway frame -> sidewalk dominant
        dict(img="a", img_num=1, x1=100, y1=500, x2=160, y2=600, score=.9, space_label="sidewalk", assoc_rescue=False),
        dict(img="b", img_num=2, x1=300, y1=500, x2=360, y2=600, score=.9, space_label="sidewalk", assoc_rescue=False),
        dict(img="c", img_num=3, x1=500, y1=500, x2=560, y2=600, score=.9, space_label="roadway", assoc_rescue=False),
    ]).assign(rider_id=0)
    r = rrc.summarize_riders(det, flow=None, params=rrc.AssocParams())
    check("dominant = majority facility", r.dominant_space.iloc[0] == "sidewalk")

    det2 = det.copy(); det2["space_label"] = ["bike_lane", "sidewalk", "roadway"]
    r2 = rrc.summarize_riders(det2, flow=None, params=rrc.AssocParams())
    check("tie broken by priority (bike lane)", r2.dominant_space.iloc[0] == "bike_lane")

    det3 = det.copy(); det3["space_label"] = ["crosswalk", "crosswalk", "sidewalk"]
    r3 = rrc.summarize_riders(det3, flow=None, params=rrc.AssocParams())
    check("crosswalk maps to roadway", r3.dominant_space.iloc[0] == "roadway")


def test_pair_diagnostic():
    import tempfile
    d = Path(tempfile.mkdtemp())
    pd.DataFrame([
        # single obs, neighbour capture 2s later exists -> pair_missed
        dict(rider_id=0, n_obs=1, assoc_rescue=False, img_num_first=5,
             direction_displacement="unknown"),
        # single obs, no neighbour within 4s -> no_pair_on_disk
        dict(rider_id=1, n_obs=1, assoc_rescue=False, img_num_first=10,
             direction_displacement="unknown"),
        # two frames but tiny displacement -> too_little_motion
        dict(rider_id=2, n_obs=2, assoc_rescue=False, img_num_first=20,
             direction_displacement="unknown"),
        # rescued pair -> rescue_distrusted
        dict(rider_id=3, n_obs=2, assoc_rescue=True, img_num_first=30,
             direction_displacement="unknown"),
        # direction known -> not audited
        dict(rider_id=4, n_obs=2, assoc_rescue=False, img_num_first=40,
             direction_displacement="along_flow"),
    ]).to_csv(d / "riders.csv", index=False)
    pd.DataFrame([
        dict(img="IM_5.jpg", img_num=5, ts=1000.0),
        dict(img="IM_6.jpg", img_num=6, ts=1002.0),     # the 2s pair
        dict(img="IM_10.jpg", img_num=10, ts=2000.0),
        dict(img="IM_11.jpg", img_num=11, ts=2600.0),   # 10 min later: not a pair
        dict(img="IM_20.jpg", img_num=20, ts=3000.0),
        dict(img="IM_30.jpg", img_num=30, ts=4000.0),
        dict(img="IM_40.jpg", img_num=40, ts=5000.0),
    ]).to_csv(d / "capture_index.csv", index=False)
    res = rrc.pair_diagnostic(d, img_dir="unused", loc_id="syn")
    check("pair diagnostic: unknown total", res["unknown"] == 4)
    check("pair diagnostic: pair_missed", res["pair_missed"] == 1)
    check("pair diagnostic: no_pair_on_disk", res["no_pair_on_disk"] == 1)
    check("pair diagnostic: too_little_motion", res["too_little_motion"] == 1)
    check("pair diagnostic: rescue_distrusted", res["rescue_distrusted"] == 1)


if __name__ == "__main__":
    test_containment_dedup()
    test_association_and_rescue()
    test_direction_rules()
    test_roi_priority()
    test_time_gap_gate()
    test_stationary_filter()
    test_dominant_space()
    test_pair_diagnostic()
    n = sum(PASS)
    print(f"\n{n}/{len(PASS)} tests passed")
    sys.exit(0 if n == len(PASS) else 1)

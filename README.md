# Bicyclist Facility-Use and Wrong-Way Riding Pipeline (v16)

Automated pipeline that measures **where cyclists ride** (bike lane / sidewalk / travel lane) and **which direction they travel** (with or against the reference flow) from motion-triggered still cameras, validated event-by-event against a complete manual count.

Data: 19 camera deployments at 17 locations in Tempe, AZ (Jan–Apr 2023). Each camera fires two still images about 2 seconds apart when motion is detected — there is no continuous video, which is what most of the design below is about.

## Headline validation (v16)

| Metric | Value |
|---|---|
| Events recovered vs. manual count | **1,823 / 1,951 = 93.4%** |
| Per-location correlation (pipeline vs. manual) | **Pearson r = 0.99** |
| Pooled wrong-way rate (pipeline, direction-known subset) | 18.3% (49/268) |
| Pooled wrong-way rate (manual, all events) | 21.8% (426/1,951) |
| Wrong-way calls visually verified | 49 / 49 |
| Wrong-way by facility (manual) | bike lane 4.6%, sidewalk 49.6% |
| Unit tests | 45 / 45 passing |

Every automated wrong-way call is exported as a side-by-side verification image and was reviewed by eye. Every filtering decision (parked bikes, exclusion zones, second-pass recoveries) is logged to an audit CSV.

## Repository layout

```
scripts/run_rider_count_new.py   # self-contained pipeline (detect → associate → direction → report)
notebooks/run_rider_count_new.ipynb  # run / inspect a single location
notebooks/batch_summary_new.ipynb    # run all locations + aggregate + manual comparison
configs/locations_new/           # per-location ROI polygons + flow arrow (loc_XX.json)
configs/locations_new/*.exclude.json # human-verified static false-positive zones
data/manual_counts_new.csv       # complete manual ground truth (19 deployments)
tests/test_pipeline_new.py       # 45 unit tests, no image data needed
```

All new work carries the `_new` suffix; the original project files are untouched. Raw images (~134 GB) are not in the repo — the pipeline expects a local data root with one folder per location.

## How it works

**Stage 1 — Event construction.** YOLOv8s at conf 0.10 / imgsz 1280 (recall on small and night targets), NMS + containment dedup. Detections are associated into rider events only if they pass three gates: frame gap ≤ 3, **EXIF time gap ≤ 30 s** (motion triggering makes consecutive image numbers minutes apart), and **HSV appearance similarity ≥ 0.30** (hue-saturation histogram plus a separate brightness histogram, compared as `min(corr_HS, corr_V)` so a white shirt is never chained to a dark jacket). Parked bicycles are removed by a stationary filter (same spot, ≥6 hits over ≥5 min, present in ≥60% of the window's captures — the density gate keeps busy chokepoints). Recurring static false positives (e.g. tree foliage at night) are removed via human-verified per-location exclusion zones.

**Stage 2 — Facility and direction.** Facility from the bottom of the bounding box against the ROI polygons, majority vote per event (`dominant_space`). Direction only from a trusted displacement: two observations **within 4 s** (2× the camera's double-shot interval) moving **≥ 40 px** (calibrated: jitter < 13 px, real riders > 50 px), classified with/against/crossing by cosine against the reference flow (gate 0.5). Chains spanning multiple triggers still count as one rider but contribute no direction — visual review showed they frequently merge two different riders. When the detector misses a rider in one image of a 2 s pair, a guided second pass re-detects at conf 0.05 and accepts only position- and appearance-matched recoveries, bound directly to the rider.

**Stage 3 — Validation.** Per-location counts vs. the manual ground truth; every wrong-way call exported as a verification image and human-reviewed; direction-unknown events decomposed by cause (`pair_diagnostic`: no pair on disk / detector missed the pair / too little motion / distrusted rescue), so coverage is a measured property, not an unexplained gap.

## Quickstart

```bash
pip install ultralytics opencv-python pandas numpy pillow

# one location
python scripts/run_rider_count_new.py \
  --loc-id loc_08 \
  --img-dir <DATA_ROOT>/loc_08 \
  --configs-dir configs/locations_new \
  --outdir outputs_new/loc_08 \
  --model yolov8s.pt --conf 0.10 --imgsz 1280

# all locations
python scripts/run_rider_count_new.py --batch --data-root <DATA_ROOT> \
  --configs-dir configs/locations_new --model yolov8s.pt --conf 0.10 --imgsz 1280

# tests (no image data required)
python tests/test_pipeline_new.py
```

Or use the notebooks: `run_rider_count_new.ipynb` for a single location (including the suspect-cluster adjudication report for new exclusion zones), `batch_summary_new.ipynb` for the full run, aggregation, and the manual-count comparison tables.

Re-running after a post-processing change does not need the YOLO pass: `--reuse-detections` replays from each location's saved `detections_raw.csv`.

## Outputs (per location, `outputs_new/loc_XX/`)

- `riders.csv` — one row per rider: `dominant_space`, `dwell_s`, `direction_displacement`, `direction_source`, `wrong_way_displacement`
- `scene_summary.json` — counts, QC funnel, all parameters
- `detections_raw.csv`, `detections_riders.csv` — per-detection records (EXIF time, appearance fingerprint, facility, rider id)
- `stationary_objects.csv`, `excluded_zone_dets.csv`, `second_pass_dets.csv` — audit logs for every removal/recovery
- `capture_index.csv` — cached EXIF times for the pair diagnostic
- `direction_check/`, `viz/`, `crops/`, `report/` — verification images and HTML report

## Final parameter set (v16)

```
model=yolov8s.pt, imgsz=1280, conf=0.10, classes={1}
nms_iou=0.70, iomin_thr=0.60
association: assoc_gap=3, max_time_gap_s=30.0, min_link_app_sim=0.30
direction:   direction_max_span_s=4.0, min_move_px=40.0, cos_gate=0.5
rescue:      trust_rescue_pair_s=0.0 (experimental, off — see CHANGELOG v8–v9)
stationary:  radius=30.0, hits=6, span_s=300.0, span_frames=50, min_density=0.60
second pass: second_pass=True, second_pass_conf=0.05
```

Every threshold is justified by camera hardware or a calibrated measurement band, never tuned against the manual counts; the manual counts are used only for post-hoc validation. The full version history (v5 → v16), including the one rolled-back experiment and why, is in [`CHANGELOG_new_pipeline.md`](CHANGELOG_new_pipeline.md).

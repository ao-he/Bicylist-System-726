# Changelog — Rider Counting Pipeline v2 (`_new` files)

This log covers version 2 of the rider-counting pipeline
(`scripts/run_rider_count_new.py`, `src/scene/roi_new.py`, and the two `_new`
notebooks), built alongside the original pipeline (v1). The v1 files
(`src/scene/roi.py`, `configs/locations/`, the original notebooks) are left
untouched; everything new lives in `_new`-suffixed files.

---

## Summary: v1 vs. v2

| | v1 (original) | v2 (current) |
|---|---|---|
| Counting unit | track-based events | rider-based (detect-per-frame + dedup) |
| Detector | YOLOv8n, conf 0.25 (default, `CONF_MIN` ineffective), imgsz 640 | YOLOv8s, conf 0.10, imgsz 1280 |
| Association | frame-gap only | frame gap + EXIF time gate + HSV appearance gate |
| Direction | symbol/sign only, min 5 obs, 8 px, any span | within-burst (≤4 s) displacement, ≥40 px, 3-way (with/against/crossing) |
| Facility label | any-involvement | any-involvement + dominant facility (majority vote) |
| False positives | none removed | parked-bike filter, human-verified exclusion zones, animal/box-swap gates |
| Validation | none | full manual count of 1,951 events; 93.4% counting recall, r = 0.99; every wrong-way call human-verified |
| Audit trail | none | every removal logged to CSV; QC funnel per location |
| Tests | none | 45 unit tests |

Headline validation (19 deployments, 17 physical locations):
counting **1,823 / 1,951 = 93.4%**, Pearson **r = 0.991**;
pooled wrong-way **18.3%** (pipeline direction-known subset) vs **21.8%** (manual);
facility-stratified wrong-way (manual): bike lane **4.6%** (55/1195), sidewalk **49.6%** (360/726).

---

## Changes from v1 to v2

Each entry below is one operation applied during v2 development, in order. Every
change is mechanistic (justified by camera hardware or a calibrated measurement),
and the manual counts were used only to validate the result afterwards, never as
a tuning target.

### 1. EXIF time-gated association
Added `max_time_gap_s = 30.0`. The cameras are motion-triggered, so consecutive
image numbers can be minutes apart; frame-gap logic alone merged two different
riders into one event. Falls back to frame-gap-only when EXIF is absent.

### 2. Stationary-object filter
A parked bicycle fires a detection at the same spot every time a passer-by trips
the shutter. Clusters hit in ≥6 distinct captures spanning ≥5 min are removed as
parked bikes. Removed detections go to `stationary_objects.csv`.

### 3. Dominant-facility labeling
Added `dominant_space` to `riders.csv` (majority-vote facility, priority
tie-break, crosswalk→roadway) plus `dwell_s`. Matches the manual count's
"where did this person mainly ride" definition, unlike the any-involvement
flags. Reports gained a QC funnel, per-facility scene stats, batch `dom_*`
columns, and a facility comparison against `data/manual_counts_new.csv`.

### 4. Density gate for the stationary filter
Added `stationary_min_density = 0.6`. At high-traffic locations hundreds of
different riders pass the same spot of a narrow bike lane, which the parked-bike
criteria matched (one location fell from 101% to 28% of its manual count). A
parked object appears in nearly every capture of its window; a busy chokepoint
spot is hit in only a scattered fraction. Clusters below the density threshold
are kept as riders.

### 5. Detection replay mode
`reuse_detections` replays a location from its saved `detections_raw.csv`
(skipping the YOLO pass) when only post-processing changed.

### 6. Two-second-pair diagnostic
`pair_diagnostic()` classifies every direction-unknown rider: `no_pair_on_disk`
(no second capture within 4 s), `pair_missed` (pair exists but the detector
missed the rider), `too_little_motion`, `rescue_distrusted`. EXIF times are
cached to `capture_index.csv`. Coverage becomes a measured property, not an
unexplained gap.

### 7. Two-second-pair rescue trust — tried, then rolled back
An experiment trusted rescued chains inside the camera's ~2 s double-shot
window. Validation caught a queue artifact: at one location the trusted pairs
came out 16/16 against flow (manual 25.7%, binomial p ~ 1e-9); at another,
23/23 with flow. In dense traffic the second shot often catches a different
rider behind the first, and the artifact's sign depends on camera geometry.
Rolled back: `trust_rescue_pair_s = 0.0` (default off), kept only as an opt-in
experimental flag with per-scene validation counters.

### 8. Human-adjudicated exclusion zones
Flickering static false positives (tree foliage detected as a bicycle at night)
fire intermittently and beat the density gate. No safe automatic threshold
exists, so: `report_suspect_clusters()` lists recurring same-spot clusters that
survived the automatic filter, a human verifies them against crops/viz, and
confirmed spots go into `configs/locations_new/<loc>.exclude.json`. Excluded
detections are logged to `excluded_zone_dets.csv`.

### 9. Verification-image wrong-way label fix
`render_direction_checks` / `render_visualizations` used `ww is True`, but
pandas hands back numpy bools and `numpy.True_ is not True`, so every
against-flow rider was exported as `OK_*.jpg` with a "with flow" banner. Counts
were always correct (they use `==`); only the exported images were mislabeled.
All three checks now compare with `==`.

### 10. Three gates against human-verified false wrong-way sources
A visual review flagged 43 error cases; they decomposed into four mechanisms,
three fixed here:
- **Cross-trigger merges:** chains spanning 11–21 s (inside the 30 s association
  gate) merged two different riders → backward displacement → false wrong-way.
  `direction_max_span_s` restricts direction to a single camera burst; longer
  chains still merge for counting but stay direction-unknown.
- **Within-burst box swaps:** in multi-rider scenes the detector often boxes a
  different person in the second frame. The appearance-similarity gate
  (`min_link_app_sim = 0.30`), previously rescue-only, now applies to every
  association link.
- **Animal false positive** (a dog, jittering 27 px past the old 25 px gate):
  `min_move_px` raised to 40, still inside the calibrated empty band (jitter
  <13 px, real riders >50 px).
- (The fourth mechanism, track fragmentation, affects counting only, slightly.)

### 11. Replay mode tolerates deleted images and empty CSVs
Cached detections whose image was deleted from the folder are dropped (with a
log line) instead of crashing crop export. An empty or unreadable
`detections_raw.csv` degrades to the no-detections path. Also: `imgsz` recorded
in `scene_summary` params; batch discovery skips `*.exclude.json`.

### 12. Four-second burst window and brightness-aware fingerprints
The direction window was tightened to 4 s: a 6 s cross-trigger merge sat on the
old boundary, and the double-shot is ~2 s, so 4 s = 2× hardware margin (the same
window as the pair diagnostic). Appearance fingerprints became HSV (8×8 HS +
16-bin V) with similarity `min(corr_HS, corr_V)`: a genuine 2 s pair had swapped
boxes between a dark-jacketed rider and a white-shirted one, and an HS-only
fingerprint cannot tell them apart (achromatic clothing carries no
hue/saturation signal). Replay mode re-reads each detected image once to refresh
cached HS-only fingerprints.

### 13. Guided second-pass re-detection
For single-observation riders whose 2 s pair image exists on disk but where the
detector missed the rider in that frame, a second pass re-runs detection on just
those pair frames at conf 0.05. A recovered box is accepted only if it lands in
the study ROI outside exclusion zones, is not a duplicate, sits within burst
travel range of the rider, and matches the rider's HSV fingerprint. Recoveries
are logged to `second_pass_dets.csv` and their directions tagged
`direction_source = "second_pass"`.

### 14. Bind second-pass recoveries to their rider
Validation showed recovered pairs produced zero directions: global
re-association pushed the fast 2 s pairs into the (distrusted) rescue path. The
pairing is already verified by burst range + appearance, so the recovered
detection now carries the target `rider_id` directly and the re-association
pass is dropped.

---

## What each output file contains (per location, under `outputs_new/loc_XX/`)

- `detections_raw.csv` — every detection (with EXIF `ts` and HSV `app` fingerprint)
- `detections_riders.csv` — detections + `space_label` + `rider_id`
- `riders.csv` — one row per rider; includes `dominant_space`, `dwell_s`,
  `direction_displacement`, `direction_source`, `wrong_way_displacement`
- `scene_summary.json` — counts, QC funnel, all parameters
- `stationary_objects.csv` — detections removed as parked bikes (audit)
- `excluded_zone_dets.csv` — detections removed by human-verified zones (audit)
- `second_pass_dets.csv` — pair detections recovered by guided re-detection (audit)
- `capture_index.csv` — cached EXIF times for the pair diagnostic
- `crops/`, `crops_person/`, `viz/`, `direction_check/`, `report/`

## Final parameter set (v2)

```
model=yolov8s.pt, imgsz=1280, conf=0.10, classes={1}
nms_iou=0.70, iomin_thr=0.60
association: assoc_gap=3, max_time_gap_s=30.0, min_link_app_sim=0.30
direction:   direction_max_span_s=4.0, min_move_px=40.0, cos_gate=0.5
rescue:      trust_rescue_pair_s=0.0 (experimental, off)
stationary:  radius=30.0, hits=6, span_s=300.0, span_frames=50, min_density=0.60
second pass: second_pass=True, second_pass_conf=0.05
reuse_detections=True
```

Tests: `python tests/test_pipeline_new.py` — 45/45 passing.

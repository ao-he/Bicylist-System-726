# Changelog — Rider Counting Pipeline (`_new`)

This log covers the new rider-counting pipeline (`scripts/run_rider_count_new.py`,
`src/scene/roi_new.py`, and the two `_new` notebooks) built on top of the original
repository. The original files (`src/scene/roi.py`, `configs/locations/`, the
original notebooks) are left untouched; every change here lives in `_new`-suffixed
files.

Branch: `claude/file-reading-b1hvoa`

---

## Summary: original pipeline vs. current (v16)

| | Original | Current (v16) |
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

## Version history

### v5 — Mechanistic accuracy fixes (baseline of the `_new` pipeline)
- **EXIF time-gated association** (`max_time_gap_s = 30.0`). Motion-triggered
  cameras make consecutive image numbers minutes apart; frame-gap logic alone
  merged two different riders into one event. Falls back to frame-gap-only when
  EXIF is absent.
- **Stationary-object filter.** A parked bicycle fires a detection at the same
  spot every time a passer-by trips the shutter. Clusters hit in ≥6 distinct
  captures spanning ≥5 min are removed as parked bikes. Removed detections go to
  `stationary_objects.csv`.
- **`dominant_space`** added to `riders.csv` (majority-vote facility, priority
  tie-break, crosswalk→roadway) plus `dwell_s`. Matches the manual count's
  "where did this person mainly ride" definition, unlike the any-involvement
  flags.
- Reports: funnel shows stationary/EXIF stages; scene stats and riders table
  show dominant facility; batch aggregation gains `dom_*` columns and a
  `facility_comparison_table` against `data/manual_counts_new.csv`.

### v6 — Stationary-filter false-positive fix + detection replay
- **Density gate** (`stationary_min_density = 0.6`). At high-traffic locations
  hundreds of different riders pass the same spot of a narrow bike lane, which
  the parked-bike criteria matched (loc_17 fell from 101% to 28% of manual). A
  parked object appears in nearly every capture of its window; a chokepoint spot
  is hit in only a scattered fraction. Clusters below the density threshold are
  kept as riders.
- **`reuse_detections`** mode replays a location from its saved
  `detections_raw.csv` (skipping the YOLO pass) when only post-processing
  changed.

### v7 — 2-second-pair diagnostic
- **`pair_diagnostic()`** classifies every direction-unknown rider:
  `no_pair_on_disk` (no second capture within 4 s), `pair_missed` (pair exists
  but detector missed the rider), `too_little_motion`, `rescue_distrusted`.
  EXIF times cached to `capture_index.csv`. Batch notebook Cell 8 rolls it up to
  `pair_diagnostic.csv`. Coverage becomes a measured property, not an
  unexplained gap.

### v8 → v9 — 2-second-pair rescue trust (tried, then rolled back)
- v8 trusted rescued chains inside the camera's ~2 s double-shot window.
- **Rolled back in v9** (`trust_rescue_pair_s = 0.0`, default off). Validation
  caught a queue artifact: at loc_04 the trusted pairs came out 16/16 against
  flow (manual 25.7%, binomial p~1e-9); at loc_15, 23/23 with flow. In dense
  traffic the second shot often catches a different rider behind the first, and
  the artifact's sign depends on camera geometry. Kept as an opt-in experimental
  flag with per-scene validation counters.

### v10 — Human-adjudicated exclusion zones
- Flickering static false positives (tree foliage detected as a bicycle at
  night) fire intermittently and beat the density gate. No safe automatic
  threshold, so: `report_suspect_clusters()` lists recurring same-spot clusters
  that survived the automatic filter (notebook Cell 7), a human verifies them
  against crops/viz, and confirmed spots go into
  `configs/locations_new/<loc>.exclude.json`. Excluded detections logged to
  `excluded_zone_dets.csv`. (loc_10's over-count traced to one such spot at
  (1604, 302): 56 hits over ~35 h.)

### v11 — Verification-image wrong-way label fix
- `render_direction_checks` / `render_visualizations` used `ww is True`, but
  pandas hands back numpy bools and `numpy.True_ is not True`, so every
  against-flow rider was written as `OK_*.jpg` with a "with flow" banner. Counts
  were always correct (they use `==`); only the exported images were mislabeled.
  All three checks now compare with `==`.

### v12 — Human-verified false wrong-way sources (three gates)
Reviewer flagged 43 error cases; they decomposed into four mechanisms, three
fixed here:
- **Cross-trigger merges:** chains spanning 11–21 s (inside the 30 s association
  gate) merged two different riders → backward displacement → false wrong-way.
  `direction_max_span_s` restricts direction to a single camera burst; longer
  chains still merge for counting but stay direction-unknown.
- **Within-burst box swaps:** in multi-rider scenes the detector often boxes a
  different person in the second frame. The appearance-similarity gate
  (`min_link_app_sim = 0.30`), previously rescue-only, now applies to every
  association link.
- **Animal false positive** (a dog, jittering 27 px past the 25 px gate):
  `min_move_px` raised to 40, still inside the calibrated empty band (jitter
  <13 px, real riders >50 px).
- (Fourth mechanism, track fragmentation, affects counting only, slightly.)

### v13 — Reuse mode tolerates deleted images / empty CSV
- Cached detections whose image was deleted from the folder are dropped (with a
  log line) instead of crashing crop export (loc_08 after out-of-scope captures
  were pruned). Empty/unreadable `detections_raw.csv` degrades to the
  no-detections path (loc_14's 1-capture folder). Also: `imgsz` recorded in
  `scene_summary` params; batch discovery skips `*.exclude.json`.

### v14 — 4 s burst window + brightness-aware fingerprints
- **Direction window tightened to 4 s.** A 6 s cross-trigger merge sat on the
  old boundary; the double-shot is ~2 s, so 4 s = 2× hardware margin (same
  window as the pair diagnostic).
- **HSV fingerprints.** A genuine 2 s pair swapped boxes between a dark-jacketed
  rider and a white-shirted one, and the HS-only fingerprint could not tell them
  apart (achromatic clothing carries no hue/saturation signal). Fingerprints are
  now HSV (8×8 HS + 16-bin V) and similarity is `min(corr_HS, corr_V)`, so a
  shared color spike cannot mask a brightness mismatch. Reuse mode re-reads each
  detected image once to refresh cached HS-only fingerprints.

### v15 — Guided second-pass re-detection
- For single-observation riders whose 2 s pair image exists on disk but where
  the detector missed the rider in that frame, a second pass re-runs detection
  on just those pair frames at conf 0.05. A recovered box is accepted only if it
  lands in the study ROI outside exclusion zones, is not a duplicate, sits within
  burst travel range of the rider, and matches the rider's HSV fingerprint.
  Recoveries logged to `second_pass_dets.csv`; directions tagged
  `direction_source = "second_pass"` and counted separately in `scene_summary`
  for validation.

### v16 — Bind second-pass recoveries to their rider
- loc_08 validation showed recovered pairs produced zero directions: global
  re-association pushed the fast 2 s pairs into the (distrusted) rescue path.
  The pairing was already verified by burst-range + appearance, so the recovered
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

## Final parameter set (v16)

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

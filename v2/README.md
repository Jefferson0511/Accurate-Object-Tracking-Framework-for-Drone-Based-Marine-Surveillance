# v2: rebuilt on the original MOBDrone release

The v1 submission (repository root) trained on a Roboflow export of MOBDrone and
never completed a fine-tune. This directory rebuilds the work from the
authoritative Zenodo release, on modern `ultralytics`, with YOLOv10.

Status: **in progress.** What is done and what is not is stated plainly below.

## Why rebuild rather than patch v1

Three defects in the v1 data path put a ceiling on anything trained from it.

**The export was stretched, not letterboxed.** 1920x1012 source frames were resized
to 640x640 with no aspect preservation, a 1.78x horizontal squash. I verified this
across 200 images: none have letterbox bars. Ultralytics letterboxes at inference,
so every object was a different shape at train time than at test time.

**It covered 2.5% of the data with a leaky split.** 3,206 of 126,170 frames, split
randomly. At 30 fps, adjacent frames are near-duplicates, so a random split leaks
train frames into validation. MOBDrone ships an official split by *flight*
(`DJI_0915*` train, `DJI_0804*` test) which has no such leakage.

**Resolution was the binding constraint.** Median object side is 18.4 px at 640, and
63.6% of instances are "small" by COCO's definition. `life_buoy` has a 10.5 px
median, which spans barely one cell at stride 8.

| Class | Instances | Median side @640 |
|---|---|---|
| boat | 4,350 | 51.5 px |
| wood | 1,580 | 39.2 px |
| surfboard | 700 | 22.7 px |
| person | 5,434 | 15.5 px |
| life_buoy | 1,230 | 10.5 px |

## What MOBDrone does and does not provide

The dataset ships **plain MS COCO detection annotations with no track IDs.** Verified
against the dataset's own README and the annotation schema. So HOTA, MOTA and IDF1
against ground-truth identities are **not possible** on MOBDrone, and any claim
otherwise (including in v1's framing) is wrong.

Two things it does provide, which the planned deliverables are built on:

**Altitude is encoded per frame.** `DJI_0804_0001_30m_1_000000.PNG` gives flight,
sequence, altitude, optional take, and frame index. All six altitudes (10, 20, 30,
40, 50, 60 m) are present. The parser round-trips 41,073/41,073 filenames.

**Many clips contain exactly one person.** On those, a correct tracker emits exactly
one ID for the whole sequence, so every additional ID is an unambiguous ID switch.
That yields a rigorous tracking measurement with no pseudo ground truth.

## Deliverables

| | Deliverable | Needs track IDs | Status |
|---|---|---|---|
| A | Detection benchmark by altitude, official split, YOLOv10 sweep | no | data builder in progress |
| B | ID-switch counts on single-person clips | no | designed |
| C | Pseudo-GT HOTA/IDF1, labelled as pseudo-GT | n/a | optional |

Deliverable B is the one that resolves an open question from a smoke test on 150
frames of 4K footage: `yolov8l`+ByteTrack emitted 56 person IDs where
`yolov10s`+ByteTrack emitted 18. Unique ID count alone cannot distinguish "detected
more distant swimmers" from "fragmented the same swimmer 56 times." Counting IDs on
clips with exactly one person can.

Smoke test, RTX 4050 Laptop, 6 GB, 150 frames at 3840x2160 downscaled to 1280:

| Model | Tracker | FPS | Peak VRAM |
|---|---|---|---|
| yolov10s | ByteTrack | 11.8 | 0.29 GB |
| yolov10s | BoT-SORT | 7.1 | 0.29 GB |
| yolov8l | ByteTrack | 11.9 | 0.59 GB |
| yolov8l | BoT-SORT | 6.6 | 0.75 GB |

YOLOv10's NMS-free head is compatible with both trackers, which was the main
technical risk. Note that a 6x parameter difference barely moves FPS, suggesting 4K
decode rather than inference dominates throughput. That is measured properly as part
of deliverable A.

## Usage

```bash
python data/build_mobdrone.py fetch   --out DATA_DIR   # ~5.5 GB, MD5-verified
python data/build_mobdrone.py inspect --out DATA_DIR   # validate before building
python data/build_mobdrone.py build   --out DATA_DIR --stride 10
```

`fetch` pulls only `MOBDrone_videos.zip` (5.46 GB) and the 5-class annotations,
skipping the 243 GB `images.zip` since frames are regenerated from the videos.
Every file is checked against the MD5 published in the Zenodo metadata, and the
download resumes on failure. This is not defensive over-engineering: during
development a plain download of the annotation file silently truncated at 10.8%
while returning HTTP 200, which would have corrupted every label downstream.

`build` is intentionally not implemented until `inspect` confirms on real data that
video filenames inside the zip match the clip names implied by the annotations. That
mapping is the one assumption extraction rests on, and guessing at it would be worse
than failing loudly.

Data preparation is designed to run on Kaggle, where the download takes minutes.
Measured local throughput to Zenodo was 0.23 MB/s, about 6.6 hours for the video zip.

## Licensing

v2 depends on `ultralytics` as a package rather than vendoring it. Current
`ultralytics` is **AGPL-3.0**, not the GPL-3.0 of the 8.0.3 tree vendored at the
repository root, so work in this directory is AGPL-3.0. MOBDrone itself is CC BY 4.0;
see the root [NOTICE](../NOTICE) for full attribution.

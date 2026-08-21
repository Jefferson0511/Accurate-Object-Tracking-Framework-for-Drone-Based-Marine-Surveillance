"""Build YOLO detection and tracking datasets from the original MOBDrone release.

Why this exists
---------------
The v1 submission trained on a Roboflow export of MOBDrone that had two defects:
it was resized to 640x640 by *stretching* 1920x1012 (a 1.78x horizontal squash,
which mismatches the letterboxing that ultralytics applies at inference), and it
covered only 3,206 of the 126,170 frames with a random train/val split that leaks
near-duplicate frames across the boundary.

This module builds the dataset from the authoritative Zenodo release instead:

  * frames stay at native 1920x1012 and are never pre-resized, so ultralytics
    letterboxes once at train time and train/inference geometry agree;
  * the official split is used (test = DJI_0804*, train = DJI_0915*), which
    splits by flight rather than by frame, so there is no duplicate leakage;
  * altitude is parsed out of each filename, enabling a per-altitude breakdown;
  * clips containing exactly one person for their whole duration are identified,
    because on those the correct tracker output is exactly one ID and every extra
    ID is an unambiguous ID switch. MOBDrone ships no track IDs, so this is what
    makes a rigorous tracking measurement possible without inventing labels.

Filename grammar, from the annotation file:

    DJI_0804_0001_30m_1_000000.PNG
    |______| |__| |_| | |____|
    flight   seq  alt  ?  frame index

Usage
-----
    python build_mobdrone.py fetch   --out DATA_DIR
    python build_mobdrone.py inspect --out DATA_DIR
    python build_mobdrone.py build   --out DATA_DIR --stride 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2

ZENODO_RECORD = "5996890"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

# Only these two files are needed. images.zip is 243 GB of frames we regenerate
# from the videos ourselves, and the other annotation files are subsets of this one.
WANTED = {
    "MOBDrone_videos.zip": "250c19fdd6fb0b5aa5fa43e5ad140306",
    "annotations_5_custom_classes.json": "11f578ff9ede473005d72ccccd0ae82b",
    "README.pdf": "69d0eeb7aaa44785d16f6ee39679822c",
}

# MOBDrone category ids are 1-based and non-COCO; YOLO wants contiguous 0-based.
# Order is fixed here so label files stay stable across rebuilds.
CLASSES = ["person", "boat", "surfboard", "wood", "life_buoy"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

# The take component is optional: both of these occur in the release.
#   DJI_0804_0001_30m_1_000000.PNG   (flight, seq, altitude, take, frame)
#   DJI_0804_0003_10m_000000.PNG     (flight, seq, altitude,       frame)
# Takes are 1-2 digits and frame indices are zero-padded to 6, so bounding the
# digit counts keeps the optional group unambiguous instead of relying on
# backtracking order.
FRAME_RE = re.compile(
    r"^(?P<flight>DJI_\d+)_(?P<seq>\d+)_(?P<alt>\d+)m"
    r"(?:_(?P<take>\d{1,2}))?"
    r"_(?P<frame>\d{4,})\.(?P<ext>\w+)$"
)


@dataclass
class FrameKey:
    """Parsed identity of one annotated frame.

    `clip` is everything before the frame index, which is the unit that maps to a
    single source video and therefore to a single tracking sequence.
    """

    flight: str
    seq: str
    altitude_m: int
    take: str | None
    frame_idx: int

    @property
    def clip(self) -> str:
        """Everything before the frame index, i.e. the name of the source video."""
        base = f"{self.flight}_{self.seq}_{self.altitude_m}m"
        return f"{base}_{self.take}" if self.take else base

    @property
    def split(self) -> str:
        """Official MOBDrone split, keyed on the flight prefix."""
        return "test" if self.flight == "DJI_0804" else "train"


@dataclass
class ClipStats:
    """Per-clip summary used to pick clean tracking sequences."""

    clip: str
    altitude_m: int
    split: str
    n_frames: int = 0
    # Max simultaneous instances of each class in any single frame of the clip.
    max_concurrent: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_boxes: int = 0
    # Frames containing at least one person, used to reject clips where the
    # person is only briefly visible and there is little track to follow.
    person_frames: int = 0


def parse_frame_name(name: str) -> FrameKey | None:
    """Parse an annotation `file_name` into its components, or None if unrecognized."""
    m = FRAME_RE.match(name)
    if not m:
        return None
    return FrameKey(
        flight=m["flight"],
        seq=m["seq"],
        altitude_m=int(m["alt"]),
        take=m["take"],
        frame_idx=int(m["frame"]),
    )


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    """Stream a file through MD5 so multi-GB inputs stay off the heap."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(out_dir: Path) -> None:
    """Download the needed Zenodo files, resuming and verifying each against MD5.

    Verification is not optional here: a plain download of the annotation file
    during development silently truncated at 10.8% while still returning HTTP 200,
    which would have corrupted every label downstream without any visible error.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    max_attempts = 5

    for name, expected_md5 in WANTED.items():
        dest = out_dir / name
        url = f"{ZENODO_API}/files/{name}/content"

        # One extra iteration past max_attempts so the final state is always
        # verified before we either accept the file or give up. Verifying at the
        # top of the loop also means an already-complete file costs one hash and
        # no network at all.
        for attempt in range(1, max_attempts + 2):
            if dest.exists():
                print(f"[{name}] verifying {dest.stat().st_size:,} bytes ...", flush=True)
                if md5_of(dest) == expected_md5:
                    print(f"[{name}] MD5 OK")
                    break
                print(f"[{name}] MD5 mismatch ({dest.stat().st_size:,} bytes on disk)")

            if attempt > max_attempts:
                raise RuntimeError(
                    f"{name}: gave up after {max_attempts} attempts; "
                    f"no file matching MD5 {expected_md5}"
                )

            print(f"[{name}] downloading, attempt {attempt}/{max_attempts} ...", flush=True)
            # -C - resumes from wherever the local file stopped, so a partial or
            # truncated file continues rather than restarting from zero.
            subprocess.run(
                ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
                 "-C", "-", url, "-o", str(dest)],
                check=False,
            )


def load_annotations(ann_path: Path) -> tuple[dict, dict[int, str]]:
    """Load the COCO annotation file and return it with an id -> class-name map."""
    with ann_path.open(encoding="utf8") as fh:
        coco = json.load(fh)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    unknown = set(cat_names.values()) - set(CLASS_TO_IDX)
    if unknown:
        raise ValueError(f"annotation file has unexpected categories: {sorted(unknown)}")
    return coco, cat_names


def summarize(coco: dict, cat_names: dict[int, str]) -> dict[str, ClipStats]:
    """Aggregate per-clip statistics: frame counts, altitude, concurrency by class."""
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    clips: dict[str, ClipStats] = {}
    skipped = 0
    for img in coco["images"]:
        key = parse_frame_name(img["file_name"])
        if key is None:
            skipped += 1
            continue
        st = clips.setdefault(
            key.clip,
            ClipStats(clip=key.clip, altitude_m=key.altitude_m, split=key.split),
        )
        st.n_frames += 1

        per_class: dict[str, int] = defaultdict(int)
        for ann in by_image.get(img["id"], []):
            per_class[cat_names[ann["category_id"]]] += 1
        st.total_boxes += sum(per_class.values())
        if per_class.get("person", 0) > 0:
            st.person_frames += 1
        for cname, n in per_class.items():
            st.max_concurrent[cname] = max(st.max_concurrent[cname], n)

    if skipped:
        print(f"warning: {skipped} filenames did not match the expected grammar")
    return clips


def inspect(out_dir: Path) -> None:
    """Print dataset structure, split sizes, altitude spread and tracking candidates.

    Run this before `build`. It validates that the video filenames inside the zip
    actually correspond to the clip names implied by the annotations, which is the
    one assumption the whole extraction step rests on.
    """
    coco, cat_names = load_annotations(out_dir / "annotations_5_custom_classes.json")
    print(f"images      : {len(coco['images']):,}")
    print(f"annotations : {len(coco['annotations']):,}")
    print(f"categories  : {sorted(cat_names.values())}\n")

    clips = summarize(coco, cat_names)

    for split in ("train", "test"):
        sub = [c for c in clips.values() if c.split == split]
        print(f"{split:<6} clips={len(sub):<4} frames={sum(c.n_frames for c in sub):>7,} "
              f"boxes={sum(c.total_boxes for c in sub):>7,}")

    print("\nframes and boxes by altitude:")
    by_alt: dict[int, list[ClipStats]] = defaultdict(list)
    for c in clips.values():
        by_alt[c.altitude_m].append(c)
    for alt in sorted(by_alt):
        sub = by_alt[alt]
        print(f"  {alt:>3}m  clips={len(sub):<4} frames={sum(c.n_frames for c in sub):>7,} "
              f"boxes={sum(c.total_boxes for c in sub):>7,}")

    chosen = select_tracking_clips(clips)
    held = sum(clips[n].n_frames for n in chosen)
    print(f"\ntracking clips (unambiguous ID-switch measurement): {len(chosen)}, "
          f"{held:,} frames")
    print(f"{'clip':<24} {'alt':>4} {'split':<6} {'frames':>7} {'person%':>8} other")
    for n in chosen:
        c = clips[n]
        pct = 100 * c.person_frames / c.n_frames
        other = sorted(k for k, v in c.max_concurrent.items() if k != "person" and v)
        print(f"{n:<24} {c.altitude_m:>3}m {c.split:<6} {c.n_frames:>7,} "
              f"{pct:>7.1f}% {other if other else ''}")
    # These are held out of detection training, so the cost is worth stating.
    train_total = sum(c.n_frames for c in clips.values() if c.split == "train")
    print(f"\nholding these out leaves {train_total - held:,} of {train_total:,} "
          f"train frames ({100 * (train_total - held) / train_total:.0f}%)")

    # The critical cross-check: do zip members line up with annotation clip names?
    zip_path = out_dir / "MOBDrone_videos.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            members = [n for n in zf.namelist() if not n.endswith("/")]
        print(f"\nvideo zip: {len(members)} members, e.g.")
        for n in members[:5]:
            print(f"  {n}")
        stems = {Path(n).stem for n in members}
        matched = sum(1 for c in clips if c in stems)
        print(f"\nclip-name match: {matched}/{len(clips)} annotation clips "
              f"have an identically-named video")
        if matched < len(clips):
            print("  -> extraction must map names explicitly; see unmatched examples:")
            for c in list(set(clips) - stems)[:5]:
                print(f"     annotation clip without exact video match: {c}")
    else:
        print("\nvideo zip not present yet; run `fetch` first")


def select_tracking_clips(
    clips: dict[str, ClipStats],
    *,
    min_frames: int = 200,
    min_person_fraction: float = 0.5,
) -> list[str]:
    """Pick clips where ID-switch counting is unambiguous.

    A clip qualifies when it never shows more than one person, shows a person in
    at least `min_person_fraction` of its frames, and is long enough to be worth
    tracking through. On such a clip the correct output is exactly one person
    track, so any additional person ID is an ID switch, measured with no pseudo
    ground truth.

    Other classes are deliberately tolerated. Trackers assign IDs per detection
    regardless of class, and the evaluator filters output to the person class
    before counting, so a boat in frame does not spoil the measurement. Requiring
    "no other class present" shrinks the pool from 12 clips to 5 for no benefit.
    """
    chosen = []
    for name, st in clips.items():
        if st.max_concurrent.get("person", 0) != 1:
            continue
        if st.n_frames < min_frames:
            continue
        if st.person_frames < min_person_fraction * st.n_frames:
            continue
        chosen.append(name)
    return sorted(chosen, key=lambda n: -clips[n].n_frames)


def coco_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, ...] | None:
    """Convert a COCO [x, y, w, h] box in pixels to YOLO normalized cx, cy, w, h.

    Returns None for degenerate boxes, which would otherwise become NaN targets.
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    cx, cy = (x + w / 2) / img_w, (y + h / 2) / img_h
    nw, nh = w / img_w, h / img_h
    # Clamp rather than drop: a box may sit a pixel outside the frame edge.
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    nw, nh = min(nw, 1.0), min(nh, 1.0)
    return cx, cy, nw, nh


def _video_for(clip: str, video_dir: Path) -> Path | None:
    """Resolve a clip name to its video file, handling the one release typo.

    Annotation clip `DJI_0915_0007_40m` corresponds to `DJI_0915_0007_40.mp4`;
    the trailing "m" is missing from that single filename in the release.
    """
    direct = video_dir / f"{clip}.mp4"
    if direct.exists():
        return direct
    if clip.endswith("m"):
        alt = video_dir / f"{clip[:-1]}.mp4"
        if alt.exists():
            return alt
    return None


def _extract(video: Path, wanted: dict[int, str], out_img: Path,
             jpeg_quality: int) -> tuple[int, int]:
    """Decode `video` sequentially, writing the frames named in `wanted`.

    Frames are read in order rather than sought. Seeking in H.264 snaps to the
    nearest keyframe, so `cap.set(POS_FRAMES, i)` can silently return a different
    frame than requested, which would mislabel data with no visible error. One
    clip in the release also has an unreadable tail, so a failed read ends the
    clip cleanly instead of aborting the build.
    """
    cap = cv2.VideoCapture(str(video))
    last = max(wanted)
    idx, written = 0, 0
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        stem = wanted.get(idx)
        if stem is not None:
            cv2.imwrite(str(out_img / f"{stem}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            written += 1
        idx += 1
    cap.release()
    return written, len(wanted) - written


def build(out_dir: Path, stride: int, test_stride: int = 5,
          jpeg_quality: int = 95) -> None:
    """Build the YOLO detection dataset and the tracking-evaluation ground truth.

    Three-way split, chosen so no stage contaminates another:

      detection train  DJI_0915 clips, minus every tracking clip
      detection test   DJI_0804 clips, the official held-out flight
      tracking eval    the single-person DJI_0915 clips, held out of training

    Every single-person clip lives in the train flight, so without holding them
    out the tracker would be evaluated on footage its detector was trained on.

    Frames are written at native 1920x1012 and never resized. Ultralytics
    letterboxes once at train time, which keeps train and inference geometry
    identical; the v1 Roboflow export instead stretched 16:9 into a square.
    """
    raw = out_dir
    coco, cat_names = load_annotations(raw / "annotations_5_custom_classes.json")
    clips = summarize(coco, cat_names)
    video_dir = raw.parent / "videos" / "video_split_fullhd"
    if not video_dir.is_dir():
        raise SystemExit(f"expected extracted videos at {video_dir}")

    track_clips = set(select_tracking_clips(clips))
    print(f"tracking clips held out of training: {len(track_clips)}")

    yolo_root = raw.parent / "yolo"
    track_root = raw.parent / "tracking"
    for split in ("train", "test"):
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (track_root / "gt").mkdir(parents=True, exist_ok=True)

    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    # Group work per clip so each video is decoded exactly once.
    per_clip: dict[str, list[dict]] = defaultdict(list)
    for img in coco["images"]:
        key = parse_frame_name(img["file_name"])
        if key is not None:
            per_clip[key.clip].append(img)

    meta_rows = ["image,clip,altitude_m,split,n_boxes"]
    track_rows = ["clip,altitude_m,video,n_frames,person_frames"]
    totals = defaultdict(int)

    for clip in sorted(per_clip):
        video = _video_for(clip, video_dir)
        if video is None:
            print(f"  SKIP {clip}: no video")
            continue
        images = sorted(per_clip[clip], key=lambda i: parse_frame_name(i["file_name"]).frame_idx)
        st = clips[clip]

        if clip in track_clips:
            # Tracking runs directly on the mp4, so no frames are extracted here;
            # only per-frame person boxes are written. Extracting all 28k frames
            # of these clips would cost ~11 GB for no gain.
            lines = []
            for img in images:
                fidx = parse_frame_name(img["file_name"]).frame_idx
                for ann in by_image.get(img["id"], []):
                    if cat_names[ann["category_id"]] != "person":
                        continue
                    x, y, w, h = ann["bbox"]
                    lines.append(f"{fidx},{x:.2f},{y:.2f},{w:.2f},{h:.2f}")
            (track_root / "gt" / f"{clip}.txt").write_text("\n".join(lines) + "\n")
            track_rows.append(f"{clip},{st.altitude_m},{video.name},"
                              f"{st.n_frames},{st.person_frames}")
            totals["track_clips"] += 1
            totals["track_boxes"] += len(lines)
            print(f"  TRACK {clip:<24} {st.n_frames:>6} frames  {len(lines):>6} person boxes")
            continue

        split = "test" if clip.startswith("DJI_0804") else "train"
        # Adjacent 30fps frames are near-duplicates, so both splits are thinned.
        # The test split is thinned less: it carries the per-altitude breakdown,
        # and sparse buckets need every box they can get. The 10m bucket holds
        # only 151 boxes dataset-wide, so it stays unthinned entirely; no stride
        # can manufacture data that is not there.
        clip_stride = stride if split == "train" else test_stride
        if st.altitude_m == 10:
            clip_stride = 1
        keep = {}
        for img in images:
            fidx = parse_frame_name(img["file_name"]).frame_idx
            if fidx % clip_stride:
                continue
            keep[fidx] = Path(img["file_name"]).stem

        if not keep:
            continue
        written, missed = _extract(video, keep, yolo_root / "images" / split, jpeg_quality)

        for img in images:
            fidx = parse_frame_name(img["file_name"]).frame_idx
            stem = keep.get(fidx)
            if stem is None or not (yolo_root / "images" / split / f"{stem}.jpg").exists():
                continue
            rows = []
            for ann in by_image.get(img["id"], []):
                conv = coco_to_yolo(ann["bbox"], img["width"], img["height"])
                if conv is None:
                    continue
                idx = CLASS_TO_IDX[cat_names[ann["category_id"]]]
                rows.append(f"{idx} " + " ".join(f"{v:.6f}" for v in conv))
            # An empty label file is meaningful: it marks a true background frame.
            (yolo_root / "labels" / split / f"{stem}.txt").write_text("\n".join(rows) + "\n")
            meta_rows.append(f"{stem},{clip},{st.altitude_m},{split},{len(rows)}")
            totals[f"{split}_boxes"] += len(rows)

        totals[f"{split}_images"] += written
        note = f"  (missed {missed} undecodable)" if missed else ""
        print(f"  {split.upper():<5} {clip:<24} {written:>5} frames{note}")

    (yolo_root / "metadata.csv").write_text("\n".join(meta_rows) + "\n")
    (track_root / "manifest.csv").write_text("\n".join(track_rows) + "\n")

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (yolo_root / "mobdrone.yaml").write_text(
        "# Generated by build_mobdrone.py. Frames are native 1920x1012 and are\n"
        "# never pre-resized; ultralytics letterboxes at train time.\n"
        f"path: {yolo_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/test\n"
        f"nc: {len(CLASSES)}\n"
        f"names:\n{names}\n"
    )

    print("\n--- build summary ---")
    for k in sorted(totals):
        print(f"  {k:<16} {totals[k]:>8,}")
    print(f"\n  dataset config : {yolo_root / 'mobdrone.yaml'}")
    print(f"  tracking GT    : {track_root / 'gt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fetch", "inspect", "build"])
    ap.add_argument("--out", type=Path, required=True, help="working data directory")
    ap.add_argument("--stride", type=int, default=10,
                    help="keep every Nth frame for detection (default 10)")
    ap.add_argument("--test-stride", type=int, default=5,
                    help="stride for the test split; thinner than train because "
                         "it carries the per-altitude breakdown (default 5)")
    ap.add_argument("--jpeg-quality", type=int, default=95,
                    help="JPEG quality; small objects suffer at low values")
    args = ap.parse_args()

    if args.stage == "fetch":
        fetch(args.out)
    elif args.stage == "inspect":
        inspect(args.out)
    else:
        build(args.out, stride=args.stride, test_stride=args.test_stride,
              jpeg_quality=args.jpeg_quality)


if __name__ == "__main__":
    main()

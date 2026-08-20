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

    @property
    def is_single_person(self) -> bool:
        """True when the clip never shows more than one person and shows no other class.

        These are the clips where an ideal tracker emits exactly one ID for the
        whole sequence, which turns ID-switch counting into a direct measurement
        rather than something requiring pseudo ground truth.
        """
        others = {k: v for k, v in self.max_concurrent.items() if k != "person"}
        return self.max_concurrent.get("person", 0) == 1 and not any(others.values())


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

    singles = sorted((c for c in clips.values() if c.is_single_person),
                     key=lambda c: -c.n_frames)
    print(f"\nsingle-person clips (unambiguous ID-switch measurement): {len(singles)}")
    for c in singles[:15]:
        print(f"  {c.clip:<34} {c.altitude_m:>3}m  {c.n_frames:>5} frames  split={c.split}")

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fetch", "inspect", "build"])
    ap.add_argument("--out", type=Path, required=True, help="working data directory")
    ap.add_argument("--stride", type=int, default=10,
                    help="keep every Nth frame for detection training (default 10)")
    args = ap.parse_args()

    if args.stage == "fetch":
        fetch(args.out)
    elif args.stage == "inspect":
        inspect(args.out)
    else:
        print("build: implemented once `inspect` confirms the clip-to-video mapping",
              file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

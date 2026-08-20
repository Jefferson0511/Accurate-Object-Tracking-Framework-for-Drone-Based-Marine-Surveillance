"""Kaggle stage 1: fetch MOBDrone from Zenodo and inspect its structure.

Runs on CPU only; no GPU quota is spent here. The point of this stage is to
answer one question that cannot be answered without the real archive: do the
video filenames inside MOBDrone_videos.zip match the clip names implied by the
annotation filenames? Frame extraction in stage 2 depends entirely on that
mapping, so it is verified before anything is built on top of it.

The repo is cloned rather than vendored into this script so there is a single
source of truth for the parsing and verification logic.
"""

import os
import shutil
import subprocess
import sys

REPO = (
    "https://github.com/Jefferson0511/"
    "Accurate-Object-Tracking-Framework-for-Drone-Based-Marine-Surveillance.git"
)
CLONE_DIR = "/kaggle/working/repo"

# The archive is 5.46 GB and /kaggle/working is both capped at 20 GB and uploaded
# verbatim as kernel output, so scratch is strongly preferred. Which scratch paths
# exist varies by image, so candidates are probed by free space rather than assumed.
SCRATCH_CANDIDATES = ["/kaggle/temp", "/tmp", "/var/tmp", "/kaggle/working"]
NEEDED_BYTES = 8 * 1024**3  # archive plus headroom to unzip a few clips


def run(cmd: list[str], *, check: bool = True, **kw) -> int:
    """Run a command, streaming output into the kernel log.

    Diagnostics pass check=False so a probe failing on an unexpected image layout
    cannot abort the run, which is exactly what killed version 1.
    """
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check, **kw).returncode


def pick_scratch() -> str:
    """Return a writable directory with room for the archive, preferring non-output."""
    print("\n--- scratch candidates ---")
    best, best_free = None, 0
    for path in SCRATCH_CANDIDATES:
        probe = path if os.path.isdir(path) else os.path.dirname(path)
        if not os.path.isdir(probe):
            print(f"  {path:<20} absent")
            continue
        try:
            free = shutil.disk_usage(probe).free
        except OSError as exc:
            print(f"  {path:<20} unreadable: {exc}")
            continue
        ok = free >= NEEDED_BYTES
        print(f"  {path:<20} free {free / 1024**3:6.1f} GB  {'usable' if ok else 'too small'}")
        # First usable candidate wins, so ordering encodes preference; only fall
        # back to a bigger-but-later path if nothing earlier qualified.
        if ok and best is None:
            best, best_free = path, free
        elif free > best_free and best is None:
            best, best_free = path, free

    if best is None:
        raise RuntimeError(f"no candidate has {NEEDED_BYTES / 1024**3:.0f} GB free")
    print(f"  -> using {best}")
    return best


def main() -> None:
    print("=" * 70)
    print("MOBDrone stage 1: fetch + inspect")
    print("=" * 70)

    run(["df", "-h"], check=False)
    print(f"\npython {sys.version.split()[0]}  cpus={os.cpu_count()}")

    data_dir = os.path.join(pick_scratch(), "mobdrone_raw")
    os.makedirs(data_dir, exist_ok=True)

    if not os.path.isdir(CLONE_DIR):
        run(["git", "clone", "--depth", "1", REPO, CLONE_DIR])

    builder_dir = os.path.join(CLONE_DIR, "v2", "data")

    # fetch is MD5-verified and resumable, so a flaky link self-heals.
    run([sys.executable, "-u", "build_mobdrone.py", "fetch", "--out", data_dir],
        cwd=builder_dir)
    run([sys.executable, "-u", "build_mobdrone.py", "inspect", "--out", data_dir],
        cwd=builder_dir)

    print("\n" + "=" * 70)
    print("stage 1 complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

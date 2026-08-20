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
import subprocess
import sys

REPO = (
    "https://github.com/Jefferson0511/"
    "Accurate-Object-Tracking-Framework-for-Drone-Based-Marine-Surveillance.git"
)
CLONE_DIR = "/kaggle/working/repo"

# /kaggle/working is persisted as kernel output and capped at 20 GB, so the
# 5.46 GB archive goes to scratch instead. Only the printed log needs to survive.
DATA_DIR = "/kaggle/temp/mobdrone_raw"


def run(cmd: list[str], **kw) -> int:
    """Run a command, streaming output into the kernel log."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw).returncode


def main() -> None:
    print("=" * 70)
    print("MOBDrone stage 1: fetch + inspect")
    print("=" * 70)

    run(["nvidia-smi", "-L"]) if os.path.exists("/proc/driver/nvidia") else print("CPU-only run")
    run(["df", "-h", "/kaggle/temp", "/kaggle/working"])

    if not os.path.isdir(CLONE_DIR):
        run(["git", "clone", "--depth", "1", REPO, CLONE_DIR])

    os.makedirs(DATA_DIR, exist_ok=True)
    builder_dir = os.path.join(CLONE_DIR, "v2", "data")

    # fetch is MD5-verified and resumable, so a flaky link self-heals.
    run([sys.executable, "-u", "build_mobdrone.py", "fetch", "--out", DATA_DIR],
        cwd=builder_dir)

    run([sys.executable, "-u", "build_mobdrone.py", "inspect", "--out", DATA_DIR],
        cwd=builder_dir)

    print("\n" + "=" * 70)
    print("stage 1 complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

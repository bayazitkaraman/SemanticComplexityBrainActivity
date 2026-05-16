"""
Download the stories selected by scan_story_candidates.py.

Input:
    configs/selected_stories.txt

The file should contain comma-separated story/task names, e.g.
    lucy,merlin,notthefallintact,pieman,...

Run:
    python scripts/download_selected_stories.py \
        --dataset-dir data/openneuro/ds002345 \
        --selected-stories configs/selected_stories.txt
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_cmd(cmd, cwd=None):
    print("\n$", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/openneuro/ds002345")
    parser.add_argument("--selected-stories", default="configs/selected_stories.txt")
    parser.add_argument("--max-subjects-per-story", type=int, default=0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    selected_path = Path(args.selected_stories)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not selected_path.exists():
        raise FileNotFoundError(f"Selected story file not found: {selected_path}")

    stories_text = selected_path.read_text(encoding="utf-8").strip()
    if not stories_text:
        raise ValueError("Selected stories file is empty.")

    cmd = [
        "python",
        "scripts/download_narratives_data.py",
        "--method",
        "datalad",
        "--target-dir",
        str(dataset_dir),
        "--stories",
        stories_text,
    ]

    if args.max_subjects_per_story > 0:
        cmd.extend(["--max-subjects-per-story", str(args.max_subjects_per_story)])

    run_cmd(cmd)


if __name__ == "__main__":
    main()

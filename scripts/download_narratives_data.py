"""
Download selected files from OpenNeuro Narratives (ds002345) into data/.

This patched version is more robust:
- uses POSIX-style relative paths for DataLad
- continues when a few DataLad files fail verification
- retries failed batches file-by-file
- writes failed paths to results/diagnostics/datalad_failed_downloads.txt

Recommended:
    python scripts/download_selected_stories.py \
        --dataset-dir data/openneuro/ds002345 \
        --selected-stories configs/selected_stories.txt
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List


DATASET_ID = "ds002345"
DATALAD_URL = f"https://github.com/OpenNeuroDatasets/{DATASET_ID}.git"
DEFAULT_STORIES = ["lucy", "merlin", "notthefallintact"]


def run_cmd(cmd: List[str], cwd: str | Path | None = None, check: bool = False) -> subprocess.CompletedProcess:
    print("\n$", " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, check=False)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)
    return proc


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def parse_story_list(text: str) -> List[str]:
    if text.lower().strip() == "all":
        return ["all"]
    return [x.strip() for x in text.split(",") if x.strip()]


def clone_datalad_dataset(target_dir: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_dir.exists() and (target_dir / ".datalad").exists():
        print(f"DataLad dataset already exists: {target_dir}")
        return

    if target_dir.exists() and any(target_dir.iterdir()):
        raise RuntimeError(
            f"{target_dir} already exists and is not an empty DataLad dataset. "
            "Move it away or choose another --target-dir."
        )

    if not command_exists("datalad"):
        raise RuntimeError(
            "DataLad was not found. Install DataLad/git-annex first, or use "
            "--method openneuro-py --full. On Windows, WSL is often easier for DataLad."
        )

    run_cmd(["datalad", "clone", DATALAD_URL, str(target_dir)], check=True)


def discover_tasks(dataset_dir: Path) -> List[str]:
    tasks = set()
    for path in dataset_dir.glob("sub-*/func/*_task-*_bold.nii.gz"):
        m = re.search(r"_task-([^_]+)", path.name)
        if m:
            tasks.add(m.group(1))
    for path in dataset_dir.glob("derivatives/**/sub-*/func/*_task-*_bold.nii.gz"):
        m = re.search(r"_task-([^_]+)", path.name)
        if m:
            tasks.add(m.group(1))
    return sorted(tasks)


def collect_static_metadata(dataset_dir: Path) -> List[Path]:
    patterns = [
        "dataset_description.json",
        "README",
        "CHANGES",
        "participants.tsv",
        "participants.json",
        "task-*.json",
        "stimuli/**",
        "stimulus/**",
        "stim/**",
    ]
    paths = []
    for pattern in patterns:
        paths.extend([p for p in dataset_dir.glob(pattern) if p.is_file()])
    return sorted(set(paths))


def score_bold_path(path: Path) -> int:
    s = str(path).lower()
    score = 0
    if "derivatives" in s:
        score += 100
    if "desc-preproc" in s:
        score += 50
    if "space-mni" in s or "mni" in s:
        score += 30
    if "bold.nii.gz" in s:
        score += 5
    if "sbref" in s:
        score -= 1000
    if "_run-1" in s:
        score += 2
    return score


def extract_subject(path: Path):
    m = re.search(r"(sub-[A-Za-z0-9]+)", str(path))
    return m.group(1) if m else None


def collect_story_paths(
    dataset_dir: Path,
    stories: List[str],
    max_subjects_per_story: int = 0,
    prefer_derivatives: bool = True,
) -> List[Path]:
    if stories == ["all"]:
        stories = discover_tasks(dataset_dir)
        print("Discovered tasks/stories:", ", ".join(stories))

    all_paths = []

    for story in stories:
        print(f"\nCollecting file paths for story/task: {story}")

        raw_bolds = list(dataset_dir.glob(f"sub-*/func/*_task-{story}*_bold.nii.gz"))
        deriv_bolds = list(dataset_dir.glob(f"derivatives/**/sub-*/func/*_task-{story}*_bold.nii.gz"))
        events = list(dataset_dir.glob(f"sub-*/func/*_task-{story}*_events.tsv"))

        bolds = deriv_bolds if (prefer_derivatives and deriv_bolds) else raw_bolds

        # Select one best BOLD per subject for the current pipeline.
        by_subject = {}
        for p in bolds:
            sub = extract_subject(p)
            if not sub:
                continue
            if sub not in by_subject or score_bold_path(p) > score_bold_path(by_subject[sub]):
                by_subject[sub] = p

        selected_subjects = sorted(by_subject)
        if max_subjects_per_story and max_subjects_per_story > 0:
            selected_subjects = selected_subjects[:max_subjects_per_story]

        selected_bolds = [by_subject[sub] for sub in selected_subjects]

        # Select matching event file per subject. Prefer run-1 if multiple exist.
        selected_events = []
        for sub in selected_subjects:
            sub_events = [ev for ev in events if extract_subject(ev) == sub]
            if sub_events:
                sub_events = sorted(sub_events, key=lambda p: (0 if "_run-1" in p.name else 1, p.name))
                selected_events.append(sub_events[0])

        print(f"  BOLD files selected: {len(selected_bolds)}")
        print(f"  Event files selected: {len(selected_events)}")

        all_paths.extend(selected_bolds)
        all_paths.extend(selected_events)

    all_paths.extend(collect_static_metadata(dataset_dir))
    return sorted(set([p for p in all_paths if p.exists() and p.is_file()]))


def write_failed_log(failed: List[str], fail_log: str):
    if not failed:
        return
    log_path = Path(fail_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(failed) + "\n", encoding="utf-8")
    print(f"\nWARNING: {len(failed)} files failed to download.")
    print(f"Failed path list saved to: {log_path}")


def datalad_get_paths(dataset_dir: Path, paths: List[Path], batch_size: int = 40, fail_log: str = "results/diagnostics/datalad_failed_downloads.txt") -> None:
    if not paths:
        print("No paths found to download.")
        return

    rel_paths = [p.relative_to(dataset_dir).as_posix() for p in paths]
    print(f"\nTotal files to request with datalad get: {len(rel_paths)}")

    failed = []

    for i in range(0, len(rel_paths), batch_size):
        batch = rel_paths[i : i + batch_size]
        print(f"\nDownloading batch {i // batch_size + 1} ({len(batch)} files)")

        proc = run_cmd(["datalad", "get", *batch], cwd=dataset_dir, check=False)

        if proc.returncode == 0:
            continue

        print("\nBatch returned an error. Retrying files individually and continuing.")
        for rel_path in batch:
            proc_one = run_cmd(["datalad", "get", rel_path], cwd=dataset_dir, check=False)
            if proc_one.returncode != 0:
                failed.append(rel_path)

    write_failed_log(failed, fail_log)

    if failed:
        print("\nThe pipeline can still continue if enough subjects remain after skipping failed files.")
        print("If many files fail on Windows, use WSL/Linux for DataLad or rerun the same command.")


def download_with_datalad(args) -> None:
    target_dir = Path(args.target_dir)
    clone_datalad_dataset(target_dir)

    stories = parse_story_list(args.stories)
    paths = collect_story_paths(
        target_dir,
        stories=stories,
        max_subjects_per_story=args.max_subjects_per_story,
        prefer_derivatives=not args.raw_bold,
    )

    datalad_get_paths(
        target_dir,
        paths,
        batch_size=args.batch_size,
        fail_log=args.fail_log,
    )

    print("\nDataLad download attempt finished.")
    print(f"Dataset folder: {target_dir}")
    print("\nNext:")
    print(f"python scripts/build_subject_list_from_openneuro.py --dataset-dir {target_dir} --stories {args.stories} --output configs/stories_all.py")


def download_with_openneuro_py(args) -> None:
    if not args.full:
        raise RuntimeError(
            "openneuro-py is used here for full dataset download only. "
            "For subset/story-specific downloads, use --method datalad."
        )

    try:
        import openneuro as on
    except ImportError as exc:
        raise RuntimeError("openneuro-py is not installed. Run: pip install openneuro-py") from exc

    target_dir = Path(args.target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading full {DATASET_ID} with openneuro-py to {target_dir}")
    on.download(dataset=DATASET_ID, target_dir=str(target_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["datalad", "openneuro-py"], default="datalad")
    parser.add_argument("--target-dir", default=f"data/openneuro/{DATASET_ID}")
    parser.add_argument("--stories", default=",".join(DEFAULT_STORIES))
    parser.add_argument("--max-subjects-per-story", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--raw-bold", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--fail-log", default="results/diagnostics/datalad_failed_downloads.txt")
    args = parser.parse_args()

    if args.method == "datalad":
        download_with_datalad(args)
    elif args.method == "openneuro-py":
        download_with_openneuro_py(args)
    else:
        raise ValueError(args.method)


if __name__ == "__main__":
    main()

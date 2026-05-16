"""
Scan OpenNeuro Narratives (ds002345) and select candidate stories objectively.

Default primary-study criteria:
- at least 20 usable subjects
- at least 5 minutes duration
- transcript/stimulus candidate found
- single-run task preferred/required by default

Why single-run by default?
--------------------------
The current analysis pipeline assumes one BOLD time series and one transcript per
subject-story. Some tasks, such as pieman, may have multiple runs for the same
story. Those are useful, but they require extra run-concatenation logic. For the
main revision analysis, it is safer to select single-run stories.

Examples
--------
python scripts/scan_story_candidates.py \
    --dataset-dir data/openneuro/ds002345 \
    --min-subjects 20 \
    --min-duration-min 5 \
    --top-n 8 \
    --output results/diagnostics/story_candidate_table.csv \
    --selected-output configs/selected_stories.txt

Allow multi-run stories if you later implement run concatenation:

python scripts/scan_story_candidates.py --allow-multirun
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_subject(path: Path):
    m = re.search(r"(sub-[A-Za-z0-9]+)", str(path))
    return m.group(1) if m else None


def parse_task(path: Path):
    m = re.search(r"_task-([^_]+)", path.name)
    return m.group(1) if m else None


def parse_run(path: Path):
    m = re.search(r"_run-([^_]+)", path.name)
    return m.group(1) if m else "single"


def score_bold(path: Path):
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
    return score


def collect_bold_by_task(dataset_dir: Path, raw_only: bool = False):
    raw = list(dataset_dir.glob("sub-*/func/*_task-*_bold.nii.gz"))
    deriv = [] if raw_only else list(dataset_dir.glob("derivatives/**/sub-*/func/*_task-*_bold.nii.gz"))

    candidates = deriv if deriv else raw

    # task -> subject -> list of bold candidates
    by_task_subject = {}

    for p in candidates:
        task = parse_task(p)
        sub = parse_subject(p)
        if not task or not sub:
            continue
        by_task_subject.setdefault(task, {}).setdefault(sub, []).append(p)

    return by_task_subject


def collect_events_by_task(dataset_dir: Path):
    events = list(dataset_dir.glob("sub-*/func/*_task-*_events.tsv"))
    by_task_subject = {}

    for p in events:
        task = parse_task(p)
        sub = parse_subject(p)
        if not task or not sub:
            continue
        by_task_subject.setdefault(task, {}).setdefault(sub, []).append(p)

    return by_task_subject


def estimate_duration_from_events(event_file: Path):
    try:
        df = pd.read_csv(event_file, sep="\t")
        if "trial_type" in df.columns:
            story_rows = df[df["trial_type"].astype(str).str.lower() == "story"]
            if not story_rows.empty and "duration" in story_rows.columns:
                return float(story_rows.iloc[0]["duration"])
        if "onset" in df.columns and "duration" in df.columns and len(df) > 0:
            return float((df["onset"] + df["duration"]).max() - df["onset"].min())
    except Exception:
        return None
    return None


def is_transcript_like(path: Path, task: str):
    name = path.name.lower()
    full = str(path).lower()

    # Avoid subject-level event files being counted as transcripts
    if "/sub-" in full.replace("\\", "/") or "\\sub-" in full:
        return False
    if name.endswith("_events.tsv"):
        return False
    if "bold" in name or "events" in name:
        return False

    task = task.lower()
    if task not in name:
        return False

    transcript_keywords = [
        "transcript", "word", "words", "text", "stim", "story", "audio"
    ]
    allowed_ext = [".txt", ".tsv", ".csv", ".json", ".TextGrid", ".textgrid"]

    return any(k in name for k in transcript_keywords) and any(str(path).endswith(ext) for ext in allowed_ext)


def has_transcript_candidate(dataset_dir: Path, task: str):
    roots = [dataset_dir / "stimuli", dataset_dir / "stimulus", dataset_dir / "stim"]
    candidates = []

    for root in roots:
        if not root.exists():
            continue
        for p in root.glob(f"**/*{task}*"):
            if p.is_file() and is_transcript_like(p, task):
                candidates.append(p)

    candidates = sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))
    if candidates:
        return True, str(candidates[0])
    return False, ""


def best_event_for_subject(event_files):
    # Prefer no run or run-1, otherwise first
    event_files = sorted(event_files, key=lambda p: (0 if "_run-1" in p.name else 1, p.name))
    return event_files[0] if event_files else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/openneuro/ds002345")
    parser.add_argument("--min-subjects", type=int, default=20)
    parser.add_argument("--min-duration-min", type=float, default=5.0)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--raw-bold", action="store_true")
    parser.add_argument("--allow-multirun", action="store_true",
                        help="Allow tasks with more than one run per subject.")
    parser.add_argument("--allow-missing-transcript", action="store_true",
                        help="Allow tasks even if no transcript/stimulus candidate is found.")
    parser.add_argument("--output", default="results/diagnostics/story_candidate_table.csv")
    parser.add_argument("--selected-output", default="configs/selected_stories.txt")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    bold_by_task = collect_bold_by_task(dataset_dir, raw_only=args.raw_bold)
    event_by_task = collect_events_by_task(dataset_dir)

    rows = []
    all_tasks = sorted(set(bold_by_task) | set(event_by_task))

    for task in all_tasks:
        bold_subjects = set(bold_by_task.get(task, {}))
        event_subjects = set(event_by_task.get(task, {}))
        usable_subjects = sorted(bold_subjects & event_subjects)

        # Count multirun subjects based on BOLD files
        n_multirun_subjects = 0
        max_bold_runs_per_subject = 0
        for sub in usable_subjects:
            n_runs = len(bold_by_task.get(task, {}).get(sub, []))
            max_bold_runs_per_subject = max(max_bold_runs_per_subject, n_runs)
            if n_runs > 1:
                n_multirun_subjects += 1

        is_single_run_task = (max_bold_runs_per_subject <= 1)

        durations = []
        for sub in usable_subjects[: min(len(usable_subjects), 20)]:
            ev = best_event_for_subject(event_by_task[task][sub])
            if ev:
                dur = estimate_duration_from_events(ev)
                if dur is not None:
                    durations.append(dur)

        median_duration_sec = float(pd.Series(durations).median()) if durations else None
        duration_min = median_duration_sec / 60.0 if median_duration_sec else None

        transcript_found, transcript_example = has_transcript_candidate(dataset_dir, task)

        eligible = (
            len(usable_subjects) >= args.min_subjects
            and duration_min is not None
            and duration_min >= args.min_duration_min
            and (args.allow_multirun or is_single_run_task)
            and (args.allow_missing_transcript or transcript_found)
        )

        rows.append({
            "story": task,
            "n_bold_subjects": len(bold_subjects),
            "n_event_subjects": len(event_subjects),
            "n_usable_subjects": len(usable_subjects),
            "median_duration_min": duration_min,
            "max_bold_runs_per_subject": max_bold_runs_per_subject,
            "n_multirun_subjects": n_multirun_subjects,
            "single_run_task": is_single_run_task,
            "transcript_candidate_found": transcript_found,
            "transcript_candidate_example": transcript_example,
            "eligible": eligible,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["eligible", "n_usable_subjects", "median_duration_min"],
            ascending=[False, False, False],
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    selected = df[df["eligible"]].head(args.top_n)["story"].tolist()

    selected_output = Path(args.selected_output)
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.write_text(",".join(selected), encoding="utf-8")

    print(f"Saved story candidate table: {output}")
    print(f"Saved selected stories: {selected_output}")
    print("\nSelected stories:")
    for s in selected:
        print(" -", s)

    if len(selected) < args.top_n:
        print(f"\nWARNING: Only {len(selected)} stories met the criteria.")
        print("You can relax criteria with --allow-multirun or --allow-missing-transcript.")

    print("\nTop candidate table:")
    cols = [
        "story", "n_usable_subjects", "median_duration_min",
        "single_run_task", "n_multirun_subjects",
        "transcript_candidate_found", "eligible"
    ]
    show_cols = [c for c in cols if c in df.columns]
    print(df[show_cols].head(max(args.top_n, 15)).to_string(index=False))


if __name__ == "__main__":
    main()

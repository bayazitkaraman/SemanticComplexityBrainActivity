"""
Build configs/stories_all.py from an OpenNeuro/DataLad BIDS download.

Patched version:
- avoids using subject events files as transcripts
- prefers single-run / run-1 files for the current pipeline
- optionally skips unreadable BOLD files
- reports skipped subjects

Example
-------
PowerShell:
    $stories = Get-Content configs/selected_stories.txt
    python scripts/build_subject_list_from_openneuro.py --dataset-dir data/openneuro/ds002345 --stories $stories --output configs/stories_all.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DEFAULT_STORIES = ["lucy", "merlin", "notthefallintact"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def parse_story_list(text: str):
    return [x.strip() for x in text.split(",") if x.strip()]


def extract_subject(path: Path):
    m = re.search(r"(sub-[A-Za-z0-9]+)", str(path))
    return m.group(1) if m else None


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
    if "_run-1" in s:
        score += 2
    return score


def bold_is_readable(path: Path):
    if not path.exists():
        return False
    try:
        if path.stat().st_size < 1024:
            return False
    except OSError:
        return False

    try:
        import nibabel as nib
        img = nib.load(str(path))
        _ = img.shape
        return True
    except Exception:
        return False


def find_bolds(dataset_dir: Path, story: str, raw_only: bool = False, require_readable: bool = True):
    raw = list(dataset_dir.glob(f"sub-*/func/*_task-{story}*_bold.nii.gz"))
    deriv = [] if raw_only else list(dataset_dir.glob(f"derivatives/**/sub-*/func/*_task-{story}*_bold.nii.gz"))

    candidates = deriv if deriv else raw
    by_subject = {}
    skipped_unreadable = []

    for p in candidates:
        sub = extract_subject(p)
        if not sub:
            continue

        if require_readable and not bold_is_readable(p):
            skipped_unreadable.append(str(p))
            continue

        if sub not in by_subject or score_bold(p) > score_bold(by_subject[sub]):
            by_subject[sub] = p

    return {sub: by_subject[sub] for sub in sorted(by_subject)}, skipped_unreadable


def find_event(dataset_dir: Path, story: str, subject: str):
    patterns = [
        f"{subject}/func/{subject}_task-{story}*_events.tsv",
        f"{subject}/**/*_task-{story}*_events.tsv",
        f"**/{subject}/func/*_task-{story}*_events.tsv",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(dataset_dir.glob(pattern))
    matches = sorted(set([p for p in matches if p.is_file()]))
    if not matches:
        return None

    # Prefer run-1 or single-run event file
    matches = sorted(matches, key=lambda p: (0 if "_run-1" in p.name else 1, p.name))
    return matches[0]


def is_transcript_like(path: Path, story: str):
    name = path.name.lower()
    full = str(path).lower().replace("\\", "/")

    if "/sub-" in full:
        return False
    if name.endswith("_events.tsv") or "events" in name or "bold" in name:
        return False
    if story.lower() not in name:
        return False

    keywords = ["transcript", "word", "words", "text", "stim", "story", "audio"]
    allowed = [".txt", ".tsv", ".csv", ".json", ".textgrid"]
    return any(k in name for k in keywords) and any(name.endswith(ext) for ext in allowed)


def find_transcript_candidates(dataset_dir: Path, story: str):
    roots = [dataset_dir / "stimuli", dataset_dir / "stimulus", dataset_dir / "stim"]
    candidates = []

    for root in roots:
        if not root.exists():
            continue
        for p in root.glob(f"**/*{story}*"):
            if p.is_file() and is_transcript_like(p, story):
                candidates.append(p)

    return sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))


def text_from_textgrid(path: Path):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    marks = re.findall(r'mark\s*=\s*"(.*?)"', raw)
    words = []
    for m in marks:
        m = m.strip()
        if not m:
            continue
        if m.lower() in ["sp", "sil", "silence", "<s>", "</s>"]:
            continue
        words.append(m)
    return " ".join(words)


def text_from_file(path: Path):
    suffix = path.suffix.lower()

    if suffix == ".textgrid":
        return text_from_textgrid(path)

    if suffix == ".txt":
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return " ".join(txt.split())

    if suffix in [".tsv", ".csv"]:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)

        preferred_cols = [
            "word", "words", "token", "text", "transcript",
            "stimulus", "sentence", "utterance"
        ]

        for col in preferred_cols:
            if col in df.columns:
                values = df[col].dropna().astype(str).tolist()
                values = [v for v in values if v.strip() and v.strip().lower() != "nan"]
                if values:
                    return " ".join(values)

        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        best_text = ""
        for col in object_cols:
            values = df[col].dropna().astype(str).tolist()
            candidate = " ".join(values)
            if len(candidate) > len(best_text):
                best_text = candidate
        return " ".join(best_text.split())

    if suffix == ".json":
        # conservative fallback
        import json
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        texts = []

        def walk(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    if k.lower() in ["word", "words", "text", "transcript", "sentence", "utterance"]:
                        texts.append(str(v))
                    else:
                        walk(v)
            elif isinstance(x, list):
                for item in x:
                    walk(item)

        walk(obj)
        return " ".join(" ".join(texts).split())

    return ""


def ensure_transcript(dataset_dir: Path, story: str, transcript_out_dir: Path):
    transcript_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcript_out_dir / f"{story}_transcript.txt"

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    candidates = find_transcript_candidates(dataset_dir, story)

    for candidate in candidates:
        try:
            txt = text_from_file(candidate)
        except Exception as exc:
            print(f"Could not read transcript candidate {candidate}: {exc}")
            continue

        if txt and len(txt.split()) > 20:
            out_path.write_text(txt, encoding="utf-8")
            print(f"Created transcript for {story}: {out_path} from {candidate}")
            return out_path

    print(f"WARNING: Could not create transcript for story '{story}'.")
    print("         Add it manually as:", out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/openneuro/ds002345")
    parser.add_argument("--stories", default=",".join(DEFAULT_STORIES))
    parser.add_argument("--output", default="configs/stories_all.py")
    parser.add_argument("--transcript-dir", default="data/transcripts")
    parser.add_argument("--max-subjects-per-story", type=int, default=0)
    parser.add_argument("--raw-bold", action="store_true")
    parser.add_argument("--allow-unreadable-bold", action="store_true",
                        help="Do not test BOLD readability before adding to config.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    stories = parse_story_list(args.stories)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    rows = []
    warnings = []

    for story in stories:
        transcript = ensure_transcript(dataset_dir, story, Path(args.transcript_dir))
        bolds, skipped_unreadable = find_bolds(
            dataset_dir,
            story,
            raw_only=args.raw_bold,
            require_readable=not args.allow_unreadable_bold,
        )

        subjects = sorted(bolds)
        if args.max_subjects_per_story and args.max_subjects_per_story > 0:
            subjects = subjects[: args.max_subjects_per_story]

        print(f"\nStory {story}: found {len(bolds)} readable subjects, using {len(subjects)}")
        if skipped_unreadable:
            print(f"  Skipped unreadable/not-downloaded BOLD candidates: {len(skipped_unreadable)}")

        for subject in subjects:
            bold = bolds[subject]
            event = find_event(dataset_dir, story, subject)
            if event is None:
                warnings.append(f"Missing event file for {story} {subject}")
                continue

            rows.append({
                "name": story,
                "subject": subject,
                "bold_file": rel(bold),
                "events_file": rel(event),
                "transcript_file": rel(transcript),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Auto-generated by scripts/build_subject_list_from_openneuro.py\n\n"
        "stories = "
        + json.dumps(rows, indent=4)
        + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved {len(rows)} rows to {output}")

    counts = {}
    for row in rows:
        counts[row["name"]] = counts.get(row["name"], 0) + 1

    print("\nCounts by story:")
    for story, count in counts.items():
        print(f" - {story}: {count}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(" -", w)


if __name__ == "__main__":
    main()

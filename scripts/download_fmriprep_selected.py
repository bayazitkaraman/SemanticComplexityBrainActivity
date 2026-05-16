from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path


def load_stories(path: str):
    spec = importlib.util.spec_from_file_location("stories_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.stories)


def git_ls_files(root: Path, pattern: str):
    proc = subprocess.run(
        ["git", "ls-files", pattern],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [x.strip() for x in proc.stdout.splitlines() if x.strip()]


def choose_bold(root: Path, subject: str, story: str):
    patterns = [
        f"{subject}/func/{subject}_task-{story}_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz",
        f"{subject}/func/{subject}_task-{story}_space-MNI152NLin6Asym_res-native_desc-preproc_bold.nii.gz",
        f"{subject}/func/{subject}_task-{story}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        f"{subject}/func/{subject}_task-{story}_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz",
        f"{subject}/func/{subject}_task-{story}_desc-preproc_bold.nii.gz",
        f"{subject}/func/*task-{story}*space-MNI152NLin2009cAsym*desc-preproc_bold.nii.gz",
        f"{subject}/func/*task-{story}*space-MNI152NLin6Asym*desc-preproc_bold.nii.gz",
        f"{subject}/func/*task-{story}*desc-preproc_bold.nii.gz",
    ]
    for pattern in patterns:
        matches = git_ls_files(root, pattern)
        if matches:
            matches = sorted(matches, key=lambda p: (("_run-" in p), p))
            return matches[0]
    return None


def choose_confounds(root: Path, subject: str, story: str):
    patterns = [
        f"{subject}/func/{subject}_task-{story}_desc-confounds_timeseries.tsv",
        f"{subject}/func/{subject}_task-{story}_desc-confounds_regressors.tsv",
        f"{subject}/func/*task-{story}*desc-confounds_timeseries.tsv",
        f"{subject}/func/*task-{story}*desc-confounds_regressors.tsv",
    ]
    for pattern in patterns:
        matches = git_ls_files(root, pattern)
        if matches:
            matches = sorted(matches, key=lambda p: (("_run-" in p), p))
            return matches[0]
    return None


def run(cmd, cwd=None):
    print("\n$", " ".join(str(x) for x in cmd))
    return subprocess.run([str(x) for x in cmd], cwd=cwd, check=False)


def datalad_get(root: Path, paths, batch_size: int, label: str, fail_log: Path):
    paths = sorted(set([p for p in paths if p]))
    print(f"\nTotal {label} files to datalad get: {len(paths)}")
    failed = []

    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        print(f"\nDownloading {label} batch {i // batch_size + 1} ({len(batch)} files)")
        proc = run(["datalad", "get", *batch], cwd=root)
        if proc.returncode == 0:
            continue

        print("Batch failed. Retrying individually.")
        for path in batch:
            proc_one = run(["datalad", "get", path], cwd=root)
            if proc_one.returncode != 0:
                failed.append(path)

    if failed:
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_log.write_text("\n".join(failed) + "\n", encoding="utf-8")
        print(f"\nWARNING: {len(failed)} {label} files failed. See {fail_log}")

    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-list", default="configs/stories_all.py")
    parser.add_argument("--fmriprep-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--download-bold", action="store_true", default=True)
    parser.add_argument("--download-confounds", action="store_true", default=True)
    parser.add_argument("--path-list-output", default="results/diagnostics/fmriprep_selected_paths.txt")
    args = parser.parse_args()

    root = Path(args.fmriprep_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    stories = load_stories(args.subject_list)
    bold_paths, confound_paths = [], []
    missing_bold, missing_confounds = [], []

    for row in stories:
        subject = row["subject"]
        story = row["name"]

        bold = choose_bold(root, subject, story)
        confounds = choose_confounds(root, subject, story)

        if bold is None:
            missing_bold.append(f"{subject} {story}")
        else:
            bold_paths.append(bold)

        if confounds is None:
            missing_confounds.append(f"{subject} {story}")
        else:
            confound_paths.append(confounds)

    out = Path(args.path_list_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# BOLD files\n" + "\n".join(sorted(set(bold_paths))) +
        "\n\n# Confounds files\n" + "\n".join(sorted(set(confound_paths))) + "\n",
        encoding="utf-8",
    )

    print(f"Matched fMRIPrep BOLD files: {len(set(bold_paths))}/{len(stories)}")
    print(f"Matched fMRIPrep confounds files: {len(set(confound_paths))}/{len(stories)}")
    print(f"Saved path list: {out}")

    if missing_bold:
        p = Path("results/diagnostics/fmriprep_missing_bold_matches.txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(missing_bold) + "\n", encoding="utf-8")
        print(f"Missing BOLD matches: {len(missing_bold)}. See {p}")

    if missing_confounds:
        p = Path("results/diagnostics/fmriprep_missing_confounds_matches.txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(missing_confounds) + "\n", encoding="utf-8")
        print(f"Missing confounds matches: {len(missing_confounds)}. See {p}")

    if args.download_bold:
        datalad_get(
            root,
            bold_paths,
            args.batch_size,
            "fMRIPrep BOLD",
            Path("results/diagnostics/fmriprep_failed_bold_downloads.txt"),
        )

    if args.download_confounds:
        datalad_get(
            root,
            confound_paths,
            args.batch_size,
            "fMRIPrep confounds",
            Path("results/diagnostics/fmriprep_failed_confounds_downloads.txt"),
        )


if __name__ == "__main__":
    main()


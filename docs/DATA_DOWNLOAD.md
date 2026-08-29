# Data and Local File Setup

This project uses the public Narratives fMRI dataset from OpenNeuro/DataLad.

```text
OpenNeuro accession: ds002345
Dataset version used for the candidate scan: 1.1.4
```

Raw fMRI data, fMRIPrep outputs, WAV files, participant-level time series, and
large intermediate files are not included in this repository.

## Included narratives

The central analysis includes all 13 intact tasks meeting the run, duration,
and participant criteria:

```text
21styear
black
bronx
forgot
merlin
milkyway
notthefallintact
piemanpni
prettymouth
shapesphysical
shapessocial
sherlock
tunnel
```

The complete candidate flow and exclusion logic are documented in
`docs/STORY_INCLUSION.md` and
`results/jnl_13_story_extension/analysis_audit.md`.

## Local subject configurations

Create a local subject list from the BIDS dataset, map it to fMRIPrep files,
and apply the reported motion thresholds:

```bash
python scripts/build_subject_list_from_openneuro.py --dataset-dir /path/to/ds002345 --stories "shapessocial,shapesphysical,notthefallintact,milkyway,bronx,forgot,black,piemanpni,prettymouth,sherlock,merlin,21styear,tunnel" --output configs/stories_all.py --raw-bold
python scripts/build_subject_list_fmriprep.py --subject-list configs/stories_all.py --fmriprep-dir /path/to/fmriprep --output configs/stories_all_fmriprep_confounds.py
python scripts/motion_qc_from_confounds.py --subject-list configs/stories_all_fmriprep_confounds.py --filtered-output configs/stories_all_fmriprep_confounds_motionfiltered.py
```

Generated configurations contain machine-specific paths and are intentionally
ignored by Git. The tracked `configs/stories_all.py` contains 572 available
subject-story rows before motion QC. The final motion-filtered configuration
contains 567 scans from 277 participants. See `configs/README.md` for per-story
counts and file roles.

## Gentle alignments and audio

Set the Gentle root before running the analysis:

Windows PowerShell:

```powershell
$env:SEMANTIC_FMRI_TIMING="gentle"
$env:SEMANTIC_FMRI_GENTLE_DIR="D:\OpenNeuroData\narratives\stimuli\gentle"
```

macOS/Linux:

```bash
export SEMANTIC_FMRI_TIMING="gentle"
export SEMANTIC_FMRI_GENTLE_DIR="/path/to/narratives/stimuli/gentle"
```

Each story directory must contain `align.json`. The rich lexical-acoustic
baseline also uses `a.wav` from the same directory. The one naming exception is
`milkyway`, which maps to the Gentle directory `milkywayoriginal`.

```text
gentle/
├── 21styear/align.json
├── black/align.json
├── bronx/align.json
├── forgot/align.json
├── merlin/align.json
├── milkywayoriginal/align.json
├── notthefallintact/align.json
├── piemanpni/align.json
├── prettymouth/align.json
├── shapesphysical/align.json
├── shapessocial/align.json
├── sherlock/align.json
└── tunnel/align.json
```

## Primary preprocessing settings

The reported analysis uses motion-QC-passed fMRIPrep MNI-space BOLD data
without additional nuisance regression or high-pass filtering:

```powershell
$env:SEMANTIC_FMRI_CONFOUND_STRATEGY="none"
$env:SEMANTIC_FMRI_HIGH_PASS="none"
```

Motion6 and motion24 are retained as sensitivity-analysis options.

## Central analysis

After preparing the rich-audio regressors and ROI caches, run:

```bash
python scripts/jnl_13_story_loso.py --gentle-root /path/to/gentle --stage all
python scripts/jnl_13_story_transfer_figure.py
python scripts/jnl_13_story_component_sensitivity.py
```

The scripts use the unified local resource directories by default. To use a
different location, specify it directly:

```bash
python scripts/jnl_13_story_loso.py --gentle-root /path/to/gentle --subject-list configs/stories_all_fmriprep_confounds_motionfiltered.py --regressor-dir /path/to/rich_audio_regressors --roi-cache /path/to/roi_cache --stage all
```

The complete command descriptions and output locations are in the main
`README.md`.

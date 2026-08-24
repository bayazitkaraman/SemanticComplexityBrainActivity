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
python scripts/build_subject_list_from_openneuro.py --dataset-dir /path/to/ds002345 --stories "21styear,black,bronx,forgot,merlin,milkyway,notthefallintact,piemanpni,prettymouth,shapesphysical,shapessocial,sherlock,tunnel" --output configs/stories_13.py
python scripts/build_subject_list_fmriprep.py --subject-list configs/stories_13.py --fmriprep-dir /path/to/fmriprep --output configs/stories_13_fmriprep_confounds.py
python scripts/motion_qc_from_confounds.py --subject-list configs/stories_13_fmriprep_confounds.py --filtered-output configs/stories_13_fmriprep_confounds_motionfiltered.py
```

Generated configurations contain machine-specific paths and are intentionally
ignored by Git. The central driver retains separate development and evaluation
configuration arguments because the analysis inputs were originally generated
in stages. For a clean rerun, the three configuration arguments may all point
to the same motion-filtered 13-story configuration; likewise, the three
regressor-directory and ROI-cache arguments may point to unified directories.

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

The short command uses the stage-specific input locations recorded as script
defaults. With unified local inputs, point each input group to the same prepared
resource:

```bash
python scripts/jnl_13_story_loso.py --gentle-root /path/to/gentle --development-config configs/stories_13_fmriprep_confounds_motionfiltered.py --initial-evaluation-config configs/stories_13_fmriprep_confounds_motionfiltered.py --new-evaluation-config configs/stories_13_fmriprep_confounds_motionfiltered.py --development-regressor-dir /path/to/rich_audio_regressors --initial-evaluation-regressor-dir /path/to/rich_audio_regressors --new-evaluation-regressor-dir /path/to/rich_audio_regressors --development-roi-cache /path/to/roi_cache --initial-evaluation-roi-cache /path/to/roi_cache --new-evaluation-roi-cache /path/to/roi_cache --stage all
```

The complete command descriptions and output locations are in the main
`README.md`. The older `run_revision_pipeline.py` workflow is retained only for
the complementary individual-scan and earlier sensitivity analyses.

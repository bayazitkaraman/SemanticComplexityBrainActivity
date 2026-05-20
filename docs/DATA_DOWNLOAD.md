# Data and Local File Setup

This project uses the public Narratives fMRI dataset from OpenNeuro/DataLad.

```text
OpenNeuro accession: ds002345
```

Raw fMRI data, fMRIPrep outputs, WAV files, and large intermediate files are not included in this repository.

## Selected stories

The final analysis uses these eight stories:

```text
black
bronx
forgot
milkyway
notthefallintact
piemanpni
shapesphysical
shapessocial
```

## Local data paths

The exact local data layout can vary depending on how the dataset was downloaded.

The analysis uses the file paths defined in:

```text
configs/stories_all.py
```

Each story/subject entry should point to the required local files, including:

```text
bold_file
events_file
transcript_file
confounds_file
```

The `confounds_file` entry is used for motion quality control and nuisance-regression sensitivity analyses when available.

## Gentle alignment files

The final analysis uses Gentle word-level forced alignments. Set the Gentle directory before running the pipeline.

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

Expected Gentle layout:

```text
gentle/
├── black/align.json
├── bronx/align.json
├── forgot/align.json
├── milkywayoriginal/align.json
├── notthefallintact/align.json
├── piemanpni/align.json
├── shapesphysical/align.json
└── shapessocial/align.json
```

The code maps the story name `milkyway` to the Gentle folder `milkywayoriginal`.

## Audio files for acoustic controls

The acoustic-control analysis uses WAV files when available. Set the audio directory before running the acoustic-control pipeline.

Windows PowerShell:

```powershell
$env:SEMANTIC_FMRI_AUDIO_DIR="data/openneuro/ds002345/stimuli"
```

macOS/Linux:

```bash
export SEMANTIC_FMRI_AUDIO_DIR="data/openneuro/ds002345/stimuli"
```

Expected WAV filenames include:

```text
black_audio.wav
bronx_audio.wav
forgot_audio.wav
milkywayoriginal_audio.wav
notthefallintact_audio.wav
piemanpni_audio.wav
shapesphysical_audio.wav
shapessocial_audio.wav
```

## Primary analysis settings

The manuscript primary analysis used motion-QC-passed fMRIPrep BOLD data without additional nuisance regression:

```powershell
$env:SEMANTIC_FMRI_CONFOUND_STRATEGY="none"
$env:SEMANTIC_FMRI_HIGH_PASS="none"
```

Motion6 and motion24 are available as sensitivity-analysis options.

## Run the pipeline

Primary analysis:

```bash
python scripts/run_revision_pipeline.py --subject-list configs/stories_all.py
python scripts/final_stats_and_tables.py
python scripts/audit_results.py
```

Acoustic-control analysis:

```bash
python scripts/run_revision_pipeline.py --subject-list configs/stories_all.py --include-acoustic
python scripts/final_stats_and_tables.py
python scripts/audit_results.py
```

The audit script should finish with:

```text
Audit finished: all expected final files were found.
```
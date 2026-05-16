# Choosing 8 Stories

Recommended design:

> 8 stories × all usable subjects, minimum 20 usable subjects per story.

Do not choose stories based on brain results. Choose them using objective dataset criteria:

1. BOLD files available.
2. Events files available.
3. At least 20 usable subjects.
4. Story duration at least 5 minutes.
5. Transcript/stimulus file available or manually recoverable.

## Workflow

### 1. Clone dataset metadata / small initial download

```bash
python scripts/download_narratives_data.py --method datalad --stories lucy --max-subjects-per-story 1
```

This creates:

```text
data/openneuro/ds002345
```

### 2. Scan candidate stories

```bash
python scripts/scan_story_candidates.py \
    --dataset-dir data/openneuro/ds002345 \
    --min-subjects 20 \
    --min-duration-min 5 \
    --top-n 8 \
    --output results/diagnostics/story_candidate_table.csv \
    --selected-output configs/selected_stories.txt
```

### 3. Review the selected stories

Open:

```text
results/diagnostics/story_candidate_table.csv
configs/selected_stories.txt
```

### 4. Download selected stories

```bash
python scripts/download_selected_stories.py \
    --dataset-dir data/openneuro/ds002345 \
    --selected-stories configs/selected_stories.txt
```

### 5. Build the final subject list

On Windows PowerShell:

```powershell
$stories = Get-Content configs/selected_stories.txt
python scripts/build_subject_list_from_openneuro.py --dataset-dir data/openneuro/ds002345 --stories $stories --output configs/stories_all.py
```

On macOS/Linux/WSL:

```bash
python scripts/build_subject_list_from_openneuro.py \
    --dataset-dir data/openneuro/ds002345 \
    --stories $(cat configs/selected_stories.txt) \
    --output configs/stories_all.py
```

### 6. Quick test

```bash
python scripts/run_revision_pipeline.py --quick --subject-list configs/stories_all.py
```

### 7. Full run

```bash
python scripts/run_revision_pipeline.py --subject-list configs/stories_all.py
```

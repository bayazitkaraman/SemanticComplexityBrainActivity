# Data download guide

The dataset used by this project is OpenNeuro `ds002345`, the Narratives dataset.

## Recommended method: DataLad

DataLad is best because it downloads only the files requested by the project.

```bash
python scripts/download_narratives_data.py --method datalad
python scripts/build_subject_list_from_openneuro.py \
    --dataset-dir data/openneuro/ds002345 \
    --stories lucy,merlin,notthefallintact \
    --output configs/stories_all.py
```

## Quick test

```bash
python scripts/download_narratives_data.py --method datalad --max-subjects-per-story 2
python scripts/build_subject_list_from_openneuro.py \
    --dataset-dir data/openneuro/ds002345 \
    --stories lucy,merlin,notthefallintact \
    --max-subjects-per-story 2 \
    --output configs/stories_quick.py
python scripts/run_revision_pipeline.py --quick --subject-list configs/stories_quick.py
```

## Full dataset with openneuro-py

```bash
pip install openneuro-py
python scripts/download_narratives_data.py --method openneuro-py --full
```

This may download a very large amount of data.

## Notes

- `data/` is ignored by GitHub.
- `configs/stories_all.py` is generated after the data are downloaded.
- If transcripts are not detected automatically, place plain text transcripts at:
  - `data/transcripts/lucy_transcript.txt`
  - `data/transcripts/merlin_transcript.txt`
  - `data/transcripts/notthefallintact_transcript.txt`

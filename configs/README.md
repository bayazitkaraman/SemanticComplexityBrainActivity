# Configuration Files

The final manuscript analyses use the same 13 intact narratives throughout.
The story list is stored in `selected_stories.txt` in the objective screening
order. Statistical inference does not depend on that order: each narrative is
the test story once in the 13 leave-one-story-out folds.

## Current configuration chain

```text
selected_stories.txt
    13 eligible intact narratives

stories_all.py
    Portable raw-BIDS configuration: 572 subject-story rows before motion QC

stories_all_fmriprep_confounds.py
    Local fMRIPrep BOLD and confound paths for the same 572 rows

stories_all_fmriprep_confounds_motionfiltered.py
    Final local analysis configuration: 567 scans from 277 participants
```

`stories_all.py` is tracked because its paths are repository-relative.
The two fMRIPrep configurations are ignored by Git because they contain
machine-specific absolute paths. They can be regenerated with the commands in
`docs/DATA_DOWNLOAD.md`.

## Final motion-QC counts

| Story | Scans |
|---|---:|
| shapessocial | 58 |
| shapesphysical | 58 |
| notthefallintact | 55 |
| milkyway | 53 |
| bronx | 47 |
| forgot | 46 |
| black | 45 |
| piemanpni | 46 |
| prettymouth | 40 |
| sherlock | 36 |
| merlin | 36 |
| 21styear | 24 |
| tunnel | 23 |
| **Total** | **567** |

Five rows fail the reported motion thresholds: `sub-229` in `shapessocial`,
`sub-233` in `shapesphysical`, `sub-334` in `notthefallintact`, `sub-287` in
`black`, and `sub-266` in `21styear`.

`stories_template.py` documents the minimum configuration schema.

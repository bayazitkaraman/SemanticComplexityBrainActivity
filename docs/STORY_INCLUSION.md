# Story Inclusion

## Eligibility criteria

The local scan of Narratives version 1.1.4 identified 19 fMRI task labels. The
analysis retained intact tasks with:

1. At least 20 usable participants before motion quality control.
2. At least 5 minutes' median duration.
3. A single BOLD run per participant.

Transcript availability was not used to rank or exclude candidate tasks because
the candidate scan did not search the external Gentle directory. All retained
stories nevertheless had the alignments and audio resources required for
modeling.

The two temporally scrambled `notthefall` controls were excluded because the
analysis concerns intact narratives. `pieman` and `schema` failed the run or
duration criteria; `slumlordreach` and `lucy` had fewer than 20 usable
participants.

## Included stories

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

After the same motion-quality-control rules were applied, these stories
contributed 567 subject-story scans from 277 unique participants.

The tracked portable configuration contains 572 rows before motion QC. Five
rows are excluded by the reported thresholds. The current configuration chain
and per-story final counts are documented in `configs/README.md`.

The analysis uses 13 leave-one-story-out folds: each story is omitted once, the
feature transform and neural encoding model are fitted to the other 12 stories,
and the complete omitted story is evaluated.

No story was selected or excluded according to the sign or strength of its
GPT-2 encoding result. The complete candidate flow is recorded in
`results/jnl_13_story_extension/analysis_audit.md` and the locked specification.

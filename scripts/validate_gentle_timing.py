from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from semantic_fmri.utils import (
    get_unique_stories,
    load_story_list,
    infer_gentle_align_json,
    load_gentle_words,
    choose_gentle_origin,
    read_story_event,
    words_to_tr_chunks_timed,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-list", default="configs/stories_all_fmriprep_confounds_motionfiltered.py")
    parser.add_argument("--tr", type=float, default=1.5)
    parser.add_argument("--output", default="results/diagnostics/gentle_timing_validation.csv")
    args = parser.parse_args()

    stories = get_unique_stories(load_story_list(args.subject_list))
    rows = []

    for story in stories:
        onset, duration = read_story_event(story["events_file"])
        align = infer_gentle_align_json(story)

        if align is None:
            rows.append({
                "story": story["name"],
                "align_json": None,
                "status": "missing",
            })
            continue

        words = load_gentle_words(align)
        origin = choose_gentle_origin(words, duration)
        chunks, n_trs, origin = words_to_tr_chunks_timed(words, duration, tr=args.tr, origin=origin)

        rel_first = words[0]["start"] - origin
        rel_last = words[-1]["end"] - origin
        nonempty_trs = sum(1 for c in chunks if c.strip())
        word_count_in_bins = sum(len(c.split()) for c in chunks)

        rows.append({
            "story": story["name"],
            "align_json": align,
            "status": "ok",
            "event_onset": onset,
            "event_duration": duration,
            "n_success_words": len(words),
            "first_word_start": words[0]["start"],
            "last_word_end": words[-1]["end"],
            "chosen_origin": origin,
            "relative_first_word": rel_first,
            "relative_last_word": rel_last,
            "relative_span": rel_last - max(0.0, rel_first),
            "n_trs": n_trs,
            "nonempty_trs": nonempty_trs,
            "empty_trs": n_trs - nonempty_trs,
            "words_assigned_to_bins": word_count_in_bins,
        })

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(df)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

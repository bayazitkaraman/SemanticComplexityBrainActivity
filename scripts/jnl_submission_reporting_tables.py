from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "results" / "jnl_shared_response_context_validation"
HELDOUT_DIR = ROOT / "results" / "jnl_heldout_story_replication"
OUTPUT_DIR = ROOT / "results" / "jnl_submission_reporting"
CANDIDATE_TABLE = ROOT / "results" / "diagnostics" / "story_candidate_table.csv"

STORY_ORDER = [
    "black",
    "bronx",
    "forgot",
    "milkyway",
    "notthefallintact",
    "piemanpni",
    "shapesphysical",
    "shapessocial",
    "merlin",
    "prettymouth",
    "sherlock",
]
ANALYSIS_ORDER = ["full_group", "participant_split_half", "leave_one_participant_out"]


def development_story_means(filename: str, analysis: str) -> pd.DataFrame:
    frame = pd.read_csv(DEVELOPMENT_DIR / filename)
    grouped = frame.groupby("story", as_index=False)[["rich_audio", "cumulative_context"]].mean()
    grouped = grouped.rename(
        columns={"rich_audio": "baseline_cv_r", "cumulative_context": "combined_cv_r"}
    )
    grouped["delta_cv_r"] = grouped["combined_cv_r"] - grouped["baseline_cv_r"]
    grouped["scope"] = "development"
    grouped["analysis"] = analysis
    return grouped


def heldout_full_group() -> pd.DataFrame:
    frame = pd.read_csv(HELDOUT_DIR / "heldout_full_group_long.csv")
    grouped = frame.groupby("story", as_index=False)[
        ["baseline_cv_r", "combined_cv_r", "delta_cv_r"]
    ].mean()
    grouped["scope"] = "heldout"
    grouped["analysis"] = "full_group"
    return grouped


def heldout_participant_means(filename: str, analysis: str) -> pd.DataFrame:
    frame = pd.read_csv(HELDOUT_DIR / filename).rename(
        columns={
            "mean_baseline_cv_r": "baseline_cv_r",
            "mean_combined_cv_r": "combined_cv_r",
            "mean_delta_cv_r": "delta_cv_r",
        }
    )
    frame["scope"] = "heldout"
    frame["analysis"] = analysis
    return frame


def pooled_leave_one_participant_out(absolute: pd.DataFrame) -> pd.DataFrame:
    values = absolute.loc[
        absolute["analysis"].eq("leave_one_participant_out"), "delta_cv_r"
    ].to_numpy(dtype=float)
    t_result = stats.ttest_1samp(values, 0.0)
    ci_low, ci_high = stats.t.interval(
        0.95,
        len(values) - 1,
        loc=float(np.mean(values)),
        scale=float(stats.sem(values)),
    )
    positives = int(np.sum(values > 0))
    return pd.DataFrame(
        [
            {
                "scope": "pooled_eleven_stories",
                "analysis": "leave_one_participant_out",
                "n_stories": len(values),
                "mean_delta_cv_r": float(np.mean(values)),
                "median_delta_cv_r": float(np.median(values)),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "n_positive_stories": positives,
                "t": float(t_result.statistic),
                "p_ttest_two_sided": float(t_result.pvalue),
                "p_wilcoxon_greater": float(
                    stats.wilcoxon(values, alternative="greater").pvalue
                ),
                "p_sign_greater": float(
                    stats.binomtest(
                        positives, len(values), 0.5, alternative="greater"
                    ).pvalue
                ),
            }
        ]
    )


def export_public_candidate_ranking() -> None:
    candidates = pd.read_csv(CANDIDATE_TABLE)
    public_columns = [
        "story",
        "n_bold_subjects",
        "n_event_subjects",
        "n_usable_subjects",
        "median_duration_min",
        "max_bold_runs_per_subject",
        "n_multirun_subjects",
        "single_run_task",
        "transcript_candidate_found",
        "eligible",
    ]
    missing = [column for column in public_columns if column not in candidates.columns]
    if missing:
        raise RuntimeError(f"Candidate table is missing columns: {missing}")
    ranking = candidates[public_columns].copy()
    ranking.insert(0, "objective_rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(HELDOUT_DIR / "candidate_story_ranking.csv", index=False)


def main() -> None:
    frames = [
        development_story_means("full_group_long.csv", "full_group"),
        development_story_means("split_half_long.csv", "participant_split_half"),
        development_story_means(
            "leave_one_participant_out_long.csv", "leave_one_participant_out"
        ),
        heldout_full_group(),
        heldout_participant_means(
            "heldout_split_half_story_means.csv", "participant_split_half"
        ),
        heldout_participant_means(
            "heldout_leave_one_participant_out_story_means.csv",
            "leave_one_participant_out",
        ),
    ]
    absolute = pd.concat(frames, ignore_index=True)
    absolute["story"] = pd.Categorical(absolute["story"], STORY_ORDER, ordered=True)
    absolute["analysis"] = pd.Categorical(
        absolute["analysis"], ANALYSIS_ORDER, ordered=True
    )
    absolute = absolute.sort_values(["analysis", "story"]).reset_index(drop=True)

    discrepancy = np.abs(
        absolute["combined_cv_r"]
        - absolute["baseline_cv_r"]
        - absolute["delta_cv_r"]
    )
    if float(discrepancy.max()) > 1e-12:
        raise RuntimeError("Absolute prediction values do not reproduce saved increments.")
    if len(absolute) != 33:
        raise RuntimeError(f"Expected 33 story-by-analysis rows, found {len(absolute)}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    absolute.to_csv(OUTPUT_DIR / "absolute_prediction_story_means.csv", index=False)
    pooled_leave_one_participant_out(absolute).to_csv(
        OUTPUT_DIR / "pooled_leave_one_participant_out_inference.csv", index=False
    )
    export_public_candidate_ranking()
    print(absolute.to_string(index=False))


if __name__ == "__main__":
    main()

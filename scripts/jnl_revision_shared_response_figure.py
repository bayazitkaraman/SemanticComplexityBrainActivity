"""Create the main story-level result figure for the JNL revision."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "results" / "jnl_shared_response_context_validation"
HELDOUT_DIR = ROOT / "results" / "jnl_heldout_story_replication"

DEVELOPMENT_ORDER = [
    "black",
    "bronx",
    "forgot",
    "milkyway",
    "notthefallintact",
    "piemanpni",
    "shapesphysical",
    "shapessocial",
]
HELDOUT_ORDER = ["merlin", "prettymouth", "sherlock"]

DISPLAY_NAMES = {
    "black": "black",
    "bronx": "bronx",
    "forgot": "forgot",
    "milkyway": "milkyway",
    "notthefallintact": "notthefallintact",
    "piemanpni": "piemanpni",
    "shapesphysical": "shapesphysical",
    "shapessocial": "shapessocial",
    "merlin": "merlin",
    "prettymouth": "prettymouth",
    "sherlock": "sherlock",
}

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
MAGENTA = "#CC79A7"
GRAY = "#747474"
LIGHT_GRAY = "#D7D7D7"


def read_development(filename: str) -> pd.DataFrame:
    table = pd.read_csv(DEVELOPMENT_DIR / filename).set_index("story")
    return table.loc[DEVELOPMENT_ORDER]


def read_heldout(filename: str, value_column: str) -> pd.Series:
    table = pd.read_csv(HELDOUT_DIR / filename).set_index("story")
    return table.loc[HELDOUT_ORDER, value_column]


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf(0.975, values.size - 1))
    return mean, mean - critical * sem, mean + critical * sem


def style_axis(axis: plt.Axes) -> None:
    axis.axhline(0.0, color="#343434", linewidth=0.9, linestyle="--", zorder=1)
    axis.grid(axis="y", color="#E7E7E7", linewidth=0.7, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", labelsize=8.5)


def add_mean_ci(axis: plt.Axes, x: float, values: np.ndarray) -> None:
    mean, low, high = mean_ci(values)
    axis.errorbar(
        x,
        mean,
        yerr=[[mean - low], [high - mean]],
        fmt="D",
        markersize=5.8,
        markerfacecolor="black",
        markeredgecolor="white",
        markeredgewidth=0.7,
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        zorder=6,
    )


def build_figure(output: Path) -> None:
    development_full = read_development("full_group_story_means.csv")
    heldout_full = read_heldout("heldout_full_group_story_means.csv", "delta_cv_r")
    heldout_split = read_heldout("heldout_split_half_story_means.csv", "mean_delta_cv_r")
    heldout_leave = read_heldout(
        "heldout_leave_one_participant_out_story_means.csv", "mean_delta_cv_r"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    figure = plt.figure(figsize=(8.2, 6.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    axis_a = figure.add_subplot(grid[0, :])
    axis_b = figure.add_subplot(grid[1, 0])
    axis_c = figure.add_subplot(grid[1, 1])

    # Panel A: all story-level full-group effects.
    all_stories = DEVELOPMENT_ORDER + HELDOUT_ORDER
    all_values = np.concatenate(
        [
            development_full["cumulative_context_additive"].to_numpy(float),
            heldout_full.to_numpy(float),
        ]
    )
    positions = np.arange(len(all_stories))
    axis_a.axvspan(7.5, 10.5, color="#F2F2F2", zorder=0)
    axis_a.vlines(positions, 0.0, all_values, color=LIGHT_GRAY, linewidth=1.2, zorder=2)
    axis_a.scatter(
        positions[:8],
        all_values[:8],
        s=46,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.7,
        label="Development stories",
        zorder=4,
    )
    axis_a.scatter(
        positions[8:],
        all_values[8:],
        s=54,
        marker="D",
        color=BLUE,
        edgecolor="white",
        linewidth=0.7,
        label="Held-out stories",
        zorder=4,
    )
    axis_a.axvline(7.5, color="#A7A7A7", linewidth=0.9)
    axis_a.set_xticks(positions)
    axis_a.set_xticklabels(
        [DISPLAY_NAMES[story] for story in all_stories], rotation=32, ha="right"
    )
    axis_a.set_ylabel(r"GPT-2 increment ($\Delta cv_r$)")
    axis_a.set_title("A  Full-group effect by narrative", loc="left", fontweight="bold")
    axis_a.legend(frameon=False, ncol=2, loc="upper left")
    axis_a.set_ylim(-0.052, 0.126)
    style_axis(axis_a)

    # Panel B: cumulative versus independent chunks in development stories.
    cumulative = development_full["cumulative_context_additive"].to_numpy(float)
    independent = development_full["independent_chunk_additive"].to_numpy(float)
    for cumulative_value, independent_value in zip(cumulative, independent, strict=True):
        axis_b.plot(
            [0, 1],
            [independent_value, cumulative_value],
            color="#B9B9B9",
            linewidth=1.0,
            alpha=0.9,
            zorder=2,
        )
    axis_b.scatter(
        np.zeros_like(independent),
        independent,
        s=32,
        color=GRAY,
        edgecolor="white",
        linewidth=0.6,
        label="Independent chunks",
        zorder=4,
    )
    axis_b.scatter(
        np.ones_like(cumulative),
        cumulative,
        s=38,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.6,
        label="Cumulative context",
        zorder=4,
    )
    add_mean_ci(axis_b, 0, independent)
    add_mean_ci(axis_b, 1, cumulative)
    axis_b.set_xticks([0, 1], ["Independent\nchunks", "Cumulative\ncontext"])
    axis_b.set_xlim(-0.45, 1.45)
    axis_b.set_ylim(-0.165, 0.225)
    axis_b.set_ylabel(r"GPT-2 increment ($\Delta cv_r$)")
    axis_b.set_title("B  Representation comparison", loc="left", fontweight="bold")
    axis_b.text(
        0.5,
        0.205,
        "paired story contrast: p = .096",
        ha="center",
        va="top",
        fontsize=8.1,
    )
    style_axis(axis_b)

    # Panel C: held-out effects under increasingly participant-specific tests.
    methods = ["Full group", "Split-half", "Leave-one-out"]
    method_x = np.arange(3)
    heldout_matrix = np.column_stack(
        [heldout_full.to_numpy(float), heldout_split.to_numpy(float), heldout_leave.to_numpy(float)]
    )
    story_colors = [GREEN, MAGENTA, BLUE]
    story_markers = ["o", "s", "D"]
    for index, story in enumerate(HELDOUT_ORDER):
        values = heldout_matrix[index]
        axis_c.plot(
            method_x,
            values,
            color=story_colors[index],
            linewidth=1.5,
            marker=story_markers[index],
            markersize=5.4,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=DISPLAY_NAMES[story],
            zorder=4,
        )
    for index in method_x:
        add_mean_ci(axis_c, float(index), heldout_matrix[:, index])
    axis_c.set_xticks(method_x, ["Full\ngroup", "Split-\nhalf", "Leave-one-\nout"])
    axis_c.set_xlim(-0.35, 2.35)
    axis_c.set_ylim(-0.012, 0.13)
    axis_c.set_ylabel(r"GPT-2 increment ($\Delta cv_r$)")
    axis_c.set_title("C  Held-out participant tests", loc="left", fontweight="bold")
    axis_c.legend(frameon=False, loc="upper right")
    style_axis(axis_c)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "final_figures" / "fig2_shared_response_results.png",
    )
    args = parser.parse_args()
    build_figure(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()

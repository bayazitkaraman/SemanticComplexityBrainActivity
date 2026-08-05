"""Create the cross-narrative result figure for the 13-story JNL revision."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "jnl_13_story_extension" / "loso"

STORY_ORDER = [
    "shapessocial",
    "shapesphysical",
    "notthefallintact",
    "milkyway",
    "bronx",
    "forgot",
    "black",
    "piemanpni",
    "prettymouth",
    "sherlock",
    "merlin",
    "21styear",
    "tunnel",
]

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#777777"
LIGHT_GRAY = "#D2D2D2"


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    interval = stats.t.interval(
        0.95, len(values) - 1, loc=mean, scale=stats.sem(values)
    )
    return mean, float(interval[0]), float(interval[1])


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#E8E8E8", linewidth=0.7, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", labelsize=8.2)


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


def load_story_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = pd.read_csv(RESULT_DIR / "cross_story_transfer_long.csv")
    shared = shared.groupby("story")[["rich_audio", "combined", "delta_r"]].mean()
    shared = shared.loc[STORY_ORDER]

    participant = pd.read_csv(
        RESULT_DIR / "cross_story_participant_transfer_long.csv"
    )
    participant = (
        participant.groupby(["story", "subject"])[
            ["rich_audio", "combined", "delta_r"]
        ]
        .mean()
        .groupby("story")
        .mean()
        .loc[STORY_ORDER]
    )
    return shared, participant


def build_figure(output: Path) -> None:
    shared, participant = load_story_tables()
    x = np.arange(len(STORY_ORDER))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure = plt.figure(figsize=(8.3, 6.1), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    axis_a = figure.add_subplot(grid[0, :])
    axis_b = figure.add_subplot(grid[1, 0])
    axis_c = figure.add_subplot(grid[1, 1])

    shared_delta = shared["delta_r"].to_numpy(float)
    colors = np.where(shared_delta >= 0, BLUE, ORANGE)
    axis_a.axhline(0, color="#343434", linewidth=0.9, linestyle="--", zorder=1)
    axis_a.vlines(x, 0, shared_delta, color=LIGHT_GRAY, linewidth=1.2, zorder=2)
    axis_a.scatter(
        x,
        shared_delta,
        s=46,
        c=colors,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    mean, low, high = mean_ci(shared_delta)
    axis_a.axhspan(low, high, color=BLUE, alpha=0.08, zorder=0)
    axis_a.axhline(mean, color=BLUE, linewidth=1.2, zorder=1)
    axis_a.set_xticks(x, STORY_ORDER, rotation=31, ha="right")
    axis_a.set_ylabel(r"GPT-2 increment ($\Delta r$)")
    axis_a.set_title(
        "A  Complete-story transfer: train on 12 narratives, test on the 13th",
        loc="left",
        fontweight="bold",
    )
    axis_a.set_ylim(-0.027, 0.050)
    axis_a.text(
        0.01,
        0.95,
        r"mean $\Delta r=.0174$, 95% CI [.0055, .0293]; 10/13 positive",
        transform=axis_a.transAxes,
        ha="left",
        va="top",
        fontsize=8.1,
    )
    style_axis(axis_a)

    baseline = shared["rich_audio"].to_numpy(float)
    combined = shared["combined"].to_numpy(float)
    for baseline_value, combined_value in zip(baseline, combined, strict=True):
        color = GREEN if combined_value >= baseline_value else LIGHT_GRAY
        axis_b.plot(
            [0, 1],
            [baseline_value, combined_value],
            color=color,
            linewidth=1.0,
            alpha=0.85,
            zorder=2,
        )
    axis_b.scatter(
        np.zeros_like(baseline), baseline, s=28, color=GRAY, edgecolor="white", zorder=4
    )
    axis_b.scatter(
        np.ones_like(combined), combined, s=34, color=BLUE, edgecolor="white", zorder=4
    )
    add_mean_ci(axis_b, 0, baseline)
    add_mean_ci(axis_b, 1, combined)
    axis_b.set_xticks([0, 1], ["Rich baseline", "Baseline + GPT-2"])
    axis_b.set_xlim(-0.45, 1.45)
    axis_b.set_ylabel("Held-out-story prediction correlation")
    axis_b.set_title("B  Absolute prediction", loc="left", fontweight="bold")
    style_axis(axis_b)

    participant_delta = participant["delta_r"].to_numpy(float)
    participant_colors = np.where(participant_delta >= 0, GREEN, ORANGE)
    axis_c.axhline(0, color="#343434", linewidth=0.9, linestyle="--", zorder=1)
    axis_c.vlines(x, 0, participant_delta, color=LIGHT_GRAY, linewidth=1.0, zorder=2)
    axis_c.scatter(
        x,
        participant_delta,
        s=38,
        c=participant_colors,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )
    participant_mean, participant_low, participant_high = mean_ci(participant_delta)
    axis_c.axhspan(
        participant_low, participant_high, color=GREEN, alpha=0.08, zorder=0
    )
    axis_c.axhline(participant_mean, color=GREEN, linewidth=1.2, zorder=1)
    axis_c.set_xticks(x, STORY_ORDER, rotation=52, ha="right")
    axis_c.set_ylabel(r"GPT-2 increment ($\Delta r$)")
    axis_c.set_title(
        "C  Story-and-participant holdout", loc="left", fontweight="bold"
    )
    axis_c.set_ylim(-0.015, 0.019)
    axis_c.text(
        0.02,
        0.95,
        r"mean $\Delta r=.0045$; 10/13 positive",
        transform=axis_c.transAxes,
        ha="left",
        va="top",
        fontsize=7.9,
    )
    style_axis(axis_c)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_DIR / "fig2_cross_story_transfer.png",
    )
    args = parser.parse_args()
    build_figure(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-table", default="results/final_tables/table4_roi_delta_stats_combined_pc1_to_pc5_vs_baseline.csv")
    parser.add_argument("--output-dir", default="results/final_figures")
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    from nilearn import datasets, plotting, image

    roi_table = Path(args.roi_table)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(roi_table)
    if "roi" not in df.columns or "mean" not in df.columns:
        raise ValueError("ROI table must contain columns: roi, mean")

    atlas = datasets.fetch_atlas_harvard_oxford("cort-prob-2mm")
    atlas_img = image.load_img(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    labels = list(atlas.labels)

    if atlas_data.ndim != 4:
        raise ValueError("Expected a 4D Harvard-Oxford probabilistic atlas.")

    stat = np.zeros(atlas_data.shape[:3], dtype=float)
    missing = []

    for _, row in df.iterrows():
        roi = str(row["roi"])
        value = float(row["mean"])
        if roi not in labels:
            missing.append(roi)
            continue

        idx = labels.index(roi)
        prob = atlas_data[..., idx]
        mask = prob > args.threshold
        stat[mask] = np.maximum(stat[mask], value)

    stat_img = image.new_img_like(atlas_img.slicer[..., 0], stat)
    vmax = float(np.nanmax(stat)) if np.nanmax(stat) > 0 else None

    # Clean stat-map version without a black title strip.
    png_stat = output_dir / "fig4_roi_delta_stat_map.png"
    pdf_stat = output_dir / "fig4_roi_delta_stat_map.pdf"

    display = plotting.plot_stat_map(
        stat_img,
        display_mode="ortho",
        cut_coords=(0, -48, 24),
        colorbar=True,
        cmap="viridis",
        threshold=0.0001,
        vmax=vmax,
        title=None,
    )
    display.savefig(str(png_stat), dpi=300)
    display.savefig(str(pdf_stat))
    display.close()

    # Optional glass-brain version, also without title.
    png_glass = output_dir / "fig4_roi_delta_brain_map.png"
    pdf_glass = output_dir / "fig4_roi_delta_brain_map.pdf"

    display = plotting.plot_glass_brain(
        stat_img,
        display_mode="lyrz",
        colorbar=True,
        plot_abs=False,
        cmap="viridis",
        threshold=0.0001,
        vmax=vmax,
        title=None,
    )
    display.savefig(str(png_glass), dpi=300)
    display.savefig(str(pdf_glass))
    display.close()

    missing_path = output_dir / "fig4_roi_delta_brain_map_missing_rois.txt"
    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")

    print("Saved:")
    print(" -", png_stat)
    print(" -", pdf_stat)
    print(" -", png_glass)
    print(" -", pdf_glass)
    if missing:
        print("WARNING: Some ROIs were not found in atlas labels. See:", missing_path)
    else:
        print("All ROI labels matched the Harvard-Oxford atlas.")


if __name__ == "__main__":
    main()

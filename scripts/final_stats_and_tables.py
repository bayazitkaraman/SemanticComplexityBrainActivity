from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_stories(path: str):
    spec = importlib.util.spec_from_file_location("stories_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.stories)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def bootstrap_ci(values, n_boot=5000, seed=42, ci=95):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    alpha = (100 - ci) / 2
    return float(np.percentile(means, alpha)), float(np.percentile(means, 100 - alpha))


def safe_ttest_1samp(values, popmean=0.0):
    try:
        from scipy import stats
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 2:
            return np.nan, np.nan
        res = stats.ttest_1samp(x, popmean)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def safe_wilcoxon_greater(values):
    try:
        from scipy import stats
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 2 or np.allclose(x, 0):
            return np.nan, np.nan
        res = stats.wilcoxon(x, alternative="greater", zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if len(pv) == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q_sorted = ranked * m / (np.arange(m) + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q_valid = np.empty_like(q_sorted)
    q_valid[order] = q_sorted
    q[valid] = q_valid
    return q


def summarize_values(values, seed=42):
    x = pd.Series(values).dropna().astype(float)
    if len(x) == 0:
        return {
            "n": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "sem": np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan,
            "t_stat": np.nan, "p_ttest_two_sided": np.nan,
            "wilcoxon_stat": np.nan, "p_wilcoxon_greater": np.nan,
        }
    ci_low, ci_high = bootstrap_ci(x.values, seed=seed)
    t_stat, p_t = safe_ttest_1samp(x.values)
    w_stat, p_w = safe_wilcoxon_greater(x.values)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "sem": float(x.sem()) if len(x) > 1 else 0.0,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "t_stat": t_stat,
        "p_ttest_two_sided": p_t,
        "wilcoxon_stat": w_stat,
        "p_wilcoxon_greater": p_w,
    }


def load_model_results(folder):
    path = Path(folder) / "encoding_model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    needed = {"story", "subject", "roi", "model", "cv_r"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def make_dataset_table(subject_list, motion_qc_file, output_dir):
    stories = load_stories(subject_list)
    df = pd.DataFrame(stories)
    rows = []
    for story, group in df.groupby("name"):
        rows.append({
            "story": story,
            "n_subject_story_rows": len(group),
            "n_unique_subjects": group["subject"].nunique() if "subject" in group else len(group),
            "bold_source": group.get("bold_source", pd.Series(["unknown"])).iloc[0],
            "has_confounds_file": bool("confounds_file" in group.columns and group["confounds_file"].notna().all()),
        })
    table = pd.DataFrame(rows).sort_values("story")
    table.loc[len(table)] = {
        "story": "TOTAL",
        "n_subject_story_rows": len(df),
        "n_unique_subjects": df["subject"].nunique() if "subject" in df else np.nan,
        "bold_source": "; ".join(sorted(df["bold_source"].dropna().unique())) if "bold_source" in df else "unknown",
        "has_confounds_file": bool("confounds_file" in df.columns and df["confounds_file"].notna().all()),
    }
    table.to_csv(output_dir / "table1_dataset_story_counts.csv", index=False)
    if motion_qc_file and Path(motion_qc_file).exists():
        qc = pd.read_csv(motion_qc_file)
        qcols = [c for c in ["mean_fd", "max_fd", "pct_fd_gt_0_5", "pct_fd_gt_0_9"] if c in qc.columns]
        if qcols:
            qc[qcols].describe().T.to_csv(output_dir / "table1b_motion_qc_summary.csv")
        if {"story", "keep_motion_qc"}.issubset(qc.columns):
            qc_counts = qc.groupby("story")["keep_motion_qc"].agg(["count", "sum"])
            qc_counts = qc_counts.rename(columns={"count": "n_before_qc", "sum": "n_after_qc"})
            qc_counts["n_excluded"] = qc_counts["n_before_qc"] - qc_counts["n_after_qc"]
            qc_counts.to_csv(output_dir / "table1c_motion_qc_counts_by_story.csv")
    return table


def model_summary_table(primary_df, output_dir):
    summary = primary_df.groupby("model")[["cv_r", "cv_r2"]].agg(["mean", "median", "std", "sem", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index().sort_values("cv_r_mean", ascending=False)
    summary.to_csv(output_dir / "table2_primary_model_comparison.csv", index=False)
    return summary


def paired_delta_tables(primary_df, output_dir, seed=42):
    idx = ["story", "subject", "roi"]
    pivot = primary_df.pivot_table(index=idx, columns="model", values="cv_r", aggfunc="mean").reset_index()
    planned = [
        ("combined_pc1_to_pc5", "baseline_only"),
        ("combined_pc1", "baseline_only"),
        ("gpt2_pc1_to_pc5", "baseline_only"),
        ("gpt2_pc1", "baseline_only"),
    ]
    rows = []
    for model, base in planned:
        if model not in pivot.columns or base not in pivot.columns:
            continue
        delta_col = f"delta_{model}_minus_{base}"
        pivot[delta_col] = pivot[model] - pivot[base]
        stats = summarize_values(pivot[delta_col], seed=seed)
        stats.update({"comparison": f"{model} - {base}", "model": model, "baseline": base})
        rows.append(stats)
    delta_summary = pd.DataFrame(rows)
    if not delta_summary.empty:
        cols = ["comparison", "model", "baseline"] + [c for c in delta_summary.columns if c not in ["comparison", "model", "baseline"]]
        delta_summary = delta_summary[cols]
    delta_summary.to_csv(output_dir / "table3_primary_paired_model_delta_stats.csv", index=False)
    pivot.to_csv(output_dir / "table3b_primary_paired_model_values.csv", index=False)
    target_delta = "delta_combined_pc1_to_pc5_minus_baseline_only"
    roi_rows = []
    if target_delta in pivot.columns:
        for roi, group in pivot.groupby("roi"):
            stats = summarize_values(group[target_delta], seed=seed)
            stats["roi"] = roi
            roi_rows.append(stats)
    roi_stats = pd.DataFrame(roi_rows)
    if not roi_stats.empty:
        roi_stats["q_wilcoxon_greater_fdr"] = bh_fdr(roi_stats["p_wilcoxon_greater"].values)
        roi_stats["q_ttest_two_sided_fdr"] = bh_fdr(roi_stats["p_ttest_two_sided"].values)
        roi_stats = roi_stats.sort_values("mean", ascending=False)
    roi_stats.to_csv(output_dir / "table4_roi_delta_stats_combined_pc1_to_pc5_vs_baseline.csv", index=False)
    return delta_summary, roi_stats, pivot


def layer_tables(layer_folder, output_dir):
    layer_path = Path(layer_folder) / "layer_overall_summary.csv"
    roi_path = Path(layer_folder) / "layer_roi_correlation_group_summary.csv"
    outputs = {}
    if layer_path.exists():
        layer = pd.read_csv(layer_path).sort_values("mean_abs_r", ascending=False)
        layer.to_csv(output_dir / "table5_layer_overall_summary.csv", index=False)
        outputs["layer"] = layer
    if roi_path.exists():
        roi = pd.read_csv(roi_path)
        if "layer" in roi.columns and "mean_abs_r" in roi.columns:
            best_layer = int(outputs["layer"].iloc[0]["layer"]) if "layer" in outputs and not outputs["layer"].empty else int(roi.sort_values("mean_abs_r", ascending=False).iloc[0]["layer"])
            roi_best = roi[roi["layer"] == best_layer].sort_values("mean_abs_r", ascending=False)
            roi_best.to_csv(output_dir / "table6_top_rois_best_layer.csv", index=False)
            outputs["roi_best_layer"] = roi_best
    return outputs


def sensitivity_tables(run_map, output_dir):
    rows, best_rows = [], []
    for run_name, folder in run_map.items():
        try:
            df = load_model_results(folder)
            grouped = df.groupby("model")[["cv_r", "cv_r2"]].mean().reset_index()
            grouped.insert(0, "run", run_name)
            rows.append(grouped)
            best_rows.append(grouped.sort_values("cv_r", ascending=False).head(1).copy())
        except Exception as e:
            rows.append(pd.DataFrame([{"run": run_name, "error": str(e)}]))
    if rows:
        sensitivity = pd.concat(rows, ignore_index=True)
        if "cv_r" in sensitivity.columns:
            sensitivity = sensitivity.sort_values(["run", "cv_r"], ascending=[True, False], na_position="last")
        sensitivity.to_csv(output_dir / "table7_sensitivity_model_comparison_by_strategy.csv", index=False)
    if best_rows:
        pd.concat(best_rows, ignore_index=True).to_csv(output_dir / "table7b_sensitivity_best_model_by_strategy.csv", index=False)


def motion_effect_check(paired, motion_qc_file, output_dir):
    if not motion_qc_file or not Path(motion_qc_file).exists():
        return None
    qc = pd.read_csv(motion_qc_file)
    if not {"story", "subject"}.issubset(qc.columns):
        return None
    target = "delta_combined_pc1_to_pc5_minus_baseline_only"
    if target not in paired.columns:
        return None
    delta_run = paired.groupby(["story", "subject"], as_index=False)[target].mean().rename(columns={target: "mean_roi_delta_cv_r"})
    merged = delta_run.merge(qc, on=["story", "subject"], how="inner")
    corr_rows = []
    try:
        from scipy import stats
        for col in ["mean_fd", "max_fd", "pct_fd_gt_0_5", "pct_fd_gt_0_9"]:
            if col in merged.columns:
                x = merged[col].astype(float)
                y = merged["mean_roi_delta_cv_r"].astype(float)
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() > 2:
                    r, p = stats.pearsonr(x[valid], y[valid])
                    rho, p_s = stats.spearmanr(x[valid], y[valid])
                else:
                    r = p = rho = p_s = np.nan
                corr_rows.append({"motion_metric": col, "n": int(valid.sum()), "pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": p_s})
    except Exception as e:
        corr_rows.append({"error": str(e)})
    merged.to_csv(output_dir / "table8b_run_level_delta_with_motion_qc.csv", index=False)
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(output_dir / "table8_motion_correlation_with_gpt2_delta.csv", index=False)
    return corr


def write_markdown_summary(output_dir, dataset_table, model_summary, delta_summary, roi_stats, layer_outputs):
    md = []
    md.append("# Final Results Summary\n")
    md.append("## Dataset\n")
    md.append(dataset_table.to_markdown(index=False))
    md.append("\n\n## Primary model comparison\n")
    md.append(model_summary.to_markdown(index=False))
    if not delta_summary.empty:
        md.append("\n\n## Planned paired model delta statistics\n")
        md.append(delta_summary.to_markdown(index=False))
    if not roi_stats.empty:
        md.append("\n\n## Top ROI deltas: combined PC1-PC5 minus baseline\n")
        cols = ["roi", "n", "mean", "median", "ci95_low", "ci95_high", "p_wilcoxon_greater", "q_wilcoxon_greater_fdr"]
        cols = [c for c in cols if c in roi_stats.columns]
        md.append(roi_stats[cols].head(17).to_markdown(index=False))
    if "layer" in layer_outputs:
        md.append("\n\n## Layer summary\n")
        md.append(layer_outputs["layer"].to_markdown(index=False))
    if "roi_best_layer" in layer_outputs:
        md.append("\n\n## Top ROIs for best layer\n")
        md.append(layer_outputs["roi_best_layer"].head(17).to_markdown(index=False))
    (output_dir / "FINAL_RESULTS_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")


def default_run_map():
    defaults = {
        "gentle_motionQC_only": "results/model_comparison_gentle_motionQC_only",
        "gentle_motion6_nohpf": "results/model_comparison_gentle_motion6_nohpf",
        "gentle_motion24_nohpf": "results/model_comparison_gentle_motion24_nohpf",
    }
    return {k: v for k, v in defaults.items() if Path(v).exists()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-list", default="configs/stories_all_fmriprep_confounds_motionfiltered.py")
    parser.add_argument("--primary-model-dir", default="results/model_comparison_gentle_motionQC_only")
    parser.add_argument("--primary-layer-dir", default="results/layer_analysis_gentle_motionQC_only")
    parser.add_argument("--motion-qc-file", default="results/diagnostics/motion_qc_summary.csv")
    parser.add_argument("--output-dir", default="results/final_tables")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = ensure_dir(args.output_dir)
    primary_df = load_model_results(args.primary_model_dir)
    dataset = make_dataset_table(args.subject_list, args.motion_qc_file, out)
    model_summary = model_summary_table(primary_df, out)
    delta_summary, roi_stats, paired = paired_delta_tables(primary_df, out, seed=args.seed)
    layer_outputs = layer_tables(args.primary_layer_dir, out)
    sensitivity_tables(default_run_map(), out)
    motion_effect_check(paired, args.motion_qc_file, out)
    write_markdown_summary(out, dataset, model_summary, delta_summary, roi_stats, layer_outputs)
    print(f"Saved final tables to: {out}")
    for p in sorted(out.glob("*")):
        print(" -", p)


if __name__ == "__main__":
    main()

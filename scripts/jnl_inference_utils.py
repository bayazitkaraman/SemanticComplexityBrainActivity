from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float).to_numpy()
    finite = np.isfinite(p)
    out = np.full_like(p, np.nan, dtype=float)
    if not finite.any():
        return pd.Series(out, index=p_values.index)
    finite_indices = np.flatnonzero(finite)
    finite_p = p[finite]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    n_values = len(ranked)
    adjusted = np.empty(n_values, dtype=float)
    running = 1.0
    for index in range(n_values - 1, -1, -1):
        running = min(running, ranked[index] * n_values / (index + 1))
        adjusted[index] = min(running, 1.0)
    finite_out = np.empty(n_values, dtype=float)
    finite_out[order] = adjusted
    out[finite_indices] = finite_out
    return pd.Series(out, index=p_values.index)


def simple_stats(values: pd.Series) -> dict[str, float]:
    clean = values.dropna().astype(float)
    n_values = int(clean.shape[0])
    if n_values < 2:
        return {
            "n": n_values,
            "mean": float(clean.mean()) if n_values else np.nan,
            "median": float(clean.median()) if n_values else np.nan,
            "std": np.nan,
            "sem": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "t_stat": np.nan,
            "p_ttest_two_sided": np.nan,
            "p_wilcoxon_greater": np.nan,
            "fraction_positive": float((clean > 0).mean()) if n_values else np.nan,
        }
    t_stat, p_t = stats.ttest_1samp(clean, 0.0)
    try:
        _, p_w = stats.wilcoxon(clean, alternative="greater")
    except ValueError:
        p_w = np.nan
    ci = stats.bootstrap(
        (clean.to_numpy(),),
        np.mean,
        confidence_level=0.95,
        n_resamples=10000,
        random_state=42,
        method="percentile",
    ).confidence_interval
    return {
        "n": n_values,
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=1)),
        "sem": float(stats.sem(clean)),
        "ci95_low": float(ci.low),
        "ci95_high": float(ci.high),
        "t_stat": float(t_stat),
        "p_ttest_two_sided": float(p_t),
        "p_wilcoxon_greater": float(p_w),
        "fraction_positive": float((clean > 0).mean()),
    }


def fit_mixed_effect(scan: pd.DataFrame, value_col: str) -> dict[str, float | bool | str]:
    data = scan[["story", "subject", value_col]].dropna().copy()
    if data.shape[0] < 10:
        return {
            "mixed_estimate": np.nan,
            "mixed_p_two_sided": np.nan,
            "mixed_p_one_sided_positive": np.nan,
            "mixed_converged": False,
            "mixed_warnings": "too_few_rows",
        }
    data = data.rename(columns={value_col: "y"})
    data["all_scans"] = "all"
    model = smf.mixedlm(
        "y ~ 1",
        data,
        groups=data["all_scans"],
        vc_formula={"subject": "0 + C(subject)", "story": "0 + C(story)"},
    )
    messages = []
    for method in ["lbfgs", "powell", "cg", "bfgs", "nm"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(reml=True, method=method, maxiter=2000, disp=False)
            messages.extend(f"{method}: {warning.message}" for warning in caught)
            if bool(result.converged):
                estimate = float(result.params["Intercept"])
                z_value = float(result.tvalues["Intercept"])
                return {
                    "mixed_estimate": estimate,
                    "mixed_se": float(result.bse["Intercept"]),
                    "mixed_z": z_value,
                    "mixed_p_two_sided": float(result.pvalues["Intercept"]),
                    "mixed_p_one_sided_positive": float(stats.norm.sf(z_value)),
                    "mixed_converged": True,
                    "mixed_warnings": " | ".join(messages),
                }
            messages.append(f"{method}: did not converge")
        except Exception as exc:  # pragma: no cover - optimizer fallback
            messages.append(f"{method}: {type(exc).__name__}: {exc}")
    return {
        "mixed_estimate": np.nan,
        "mixed_se": np.nan,
        "mixed_z": np.nan,
        "mixed_p_two_sided": np.nan,
        "mixed_p_one_sided_positive": np.nan,
        "mixed_converged": False,
        "mixed_warnings": " | ".join(messages),
    }

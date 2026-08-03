from __future__ import annotations

from collections.abc import Sequence

import numpy as np


COMMON_STORY_TR = 1.5


def cached_segment_times(
    n_timepoints: int,
    bold_tr: float,
    story_onset: float,
) -> np.ndarray:
    """Return cached BOLD-volume times relative to the exact story onset."""
    if n_timepoints <= 0:
        raise ValueError("A cached BOLD segment must contain at least one time point")
    if not np.isfinite(bold_tr) or bold_tr <= 0:
        raise ValueError(f"Invalid BOLD TR: {bold_tr}")
    if not np.isfinite(story_onset) or story_onset < 0:
        raise ValueError(f"Invalid story onset: {story_onset}")

    # extract_roi_timeseries starts each cached segment at int(onset // TR).
    segment_start = int(story_onset // bold_tr)
    onset_offset = story_onset - segment_start * bold_tr
    return np.arange(n_timepoints, dtype=float) * bold_tr - onset_offset


def common_story_times(
    source_times: Sequence[np.ndarray],
    common_tr: float = COMMON_STORY_TR,
    max_timepoints: int | None = None,
) -> np.ndarray:
    """Construct the non-extrapolated story grid shared by all source series."""
    if not source_times:
        raise ValueError("At least one source time vector is required")
    if not np.isfinite(common_tr) or common_tr <= 0:
        raise ValueError(f"Invalid common TR: {common_tr}")

    for times in source_times:
        if len(times) == 0 or not np.all(np.isfinite(times)):
            raise ValueError("Source time vectors must be nonempty and finite")
        if np.any(np.diff(times) <= 0):
            raise ValueError("Source times must be strictly increasing")
        if times[0] > 1e-10:
            raise ValueError("A cached BOLD segment does not support story time zero")

    minimum_supported_end = min(float(times[-1]) for times in source_times)
    n_timepoints = int(np.floor((minimum_supported_end + 1e-10) / common_tr)) + 1
    if max_timepoints is not None:
        if max_timepoints <= 0:
            raise ValueError("max_timepoints must be positive")
        n_timepoints = min(n_timepoints, int(max_timepoints))
    if n_timepoints <= 0:
        raise ValueError("No common story-grid time points are supported")
    return np.arange(n_timepoints, dtype=float) * common_tr


def interpolate_target_matrix(
    target: np.ndarray,
    source_times: np.ndarray,
    destination_times: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate all target columns without extrapolation."""
    target = np.asarray(target, dtype=float)
    if target.ndim != 2:
        raise ValueError(f"Target matrix must be two-dimensional, got {target.shape}")
    if len(target) != len(source_times):
        raise ValueError(
            f"Target/time length mismatch: {len(target)} != {len(source_times)}"
        )
    if destination_times[0] < source_times[0] - 1e-10:
        raise ValueError("Destination grid begins before BOLD support")
    if destination_times[-1] > source_times[-1] + 1e-10:
        raise ValueError("Destination grid ends after BOLD support")

    return np.column_stack(
        [
            np.interp(destination_times, source_times, target[:, column])
            for column in range(target.shape[1])
        ]
    )


def align_target_to_story_grid(
    target: np.ndarray,
    bold_tr: float,
    story_onset: float,
    common_tr: float = COMMON_STORY_TR,
    max_timepoints: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map one cached participant target to the common story-relative grid."""
    source = cached_segment_times(len(target), bold_tr, story_onset)
    destination = common_story_times(
        [source],
        common_tr=common_tr,
        max_timepoints=max_timepoints,
    )
    return interpolate_target_matrix(target, source, destination), destination


def align_targets_to_common_story_grid(
    targets: Sequence[np.ndarray],
    bold_trs: Sequence[float],
    story_onsets: Sequence[float],
    common_tr: float = COMMON_STORY_TR,
    max_timepoints: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Map multiple participant targets to their common supported story grid."""
    if not (len(targets) == len(bold_trs) == len(story_onsets)):
        raise ValueError("Targets, TRs, and onsets must have the same length")
    sources = [
        cached_segment_times(len(target), tr, onset)
        for target, tr, onset in zip(targets, bold_trs, story_onsets)
    ]
    destination = common_story_times(
        sources,
        common_tr=common_tr,
        max_timepoints=max_timepoints,
    )
    aligned = [
        interpolate_target_matrix(target, source, destination)
        for target, source in zip(targets, sources)
    ]
    return aligned, destination

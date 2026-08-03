from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from jnl_story_grid import (  # noqa: E402
    align_target_to_story_grid,
    align_targets_to_common_story_grid,
    cached_segment_times,
)


class StoryGridTests(unittest.TestCase):
    def test_fractional_onset_recovers_linear_signal(self) -> None:
        source = cached_segment_times(8, bold_tr=1.5, story_onset=29.5)
        target = np.column_stack([2.0 * source + 3.0, -source + 1.0])
        aligned, destination = align_target_to_story_grid(
            target,
            bold_tr=1.5,
            story_onset=29.5,
        )
        np.testing.assert_allclose(destination, np.arange(7) * 1.5)
        np.testing.assert_allclose(aligned[:, 0], 2.0 * destination + 3.0)
        np.testing.assert_allclose(aligned[:, 1], -destination + 1.0)

    def test_mixed_trs_use_only_common_support(self) -> None:
        source_150 = cached_segment_times(269, bold_tr=1.5, story_onset=21.0)
        source_152 = cached_segment_times(265, bold_tr=1.52, story_onset=21.0)
        target_150 = np.column_stack([source_150, source_150**2])
        target_152 = np.column_stack([source_152, 4.0 * source_152])
        aligned, destination = align_targets_to_common_story_grid(
            [target_150, target_152],
            [1.5, 1.52],
            [21.0, 21.0],
        )
        self.assertEqual(len(destination), 267)
        self.assertAlmostEqual(destination[-1], 399.0)
        np.testing.assert_allclose(aligned[0][:, 0], destination)
        np.testing.assert_allclose(aligned[1][:, 1], 4.0 * destination)

    def test_design_length_caps_destination_grid(self) -> None:
        source = cached_segment_times(20, bold_tr=1.5, story_onset=0.0)
        target = np.column_stack([source, source])
        aligned, destination = align_target_to_story_grid(
            target,
            bold_tr=1.5,
            story_onset=0.0,
            max_timepoints=11,
        )
        self.assertEqual(aligned.shape, (11, 2))
        self.assertEqual(len(destination), 11)


if __name__ == "__main__":
    unittest.main()

# Rich audio controls

- `lexical_delta`: mean = 0.0007, 95% CI [-0.0093, 0.0108], p = 0.8858, fraction positive = 0.520.
- `existing_acoustic_delta`: mean = -0.0007, 95% CI [-0.0093, 0.0083], p = 0.882, fraction positive = 0.510.
- `rich_audio_delta`: mean = -0.0009, 95% CI [-0.0069, 0.0054], p = 0.7667, fraction positive = 0.532.
- `rich_minus_original_delta`: mean = -0.0017, 95% CI [-0.0087, 0.0055], p = 0.6455, fraction positive = 0.507.

Interpretation: the key line is `rich_audio_delta`. If it stays positive, GPT-2 still adds information after stronger waveform/spectral/prosodic controls. If `rich_minus_original_delta` is negative, the richer controls explain part of the original increment.

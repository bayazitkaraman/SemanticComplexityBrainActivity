# Journal of Neurolinguistics rich audio controls

- `lexical_delta`: mean = 0.0003, 95% CI [-0.0094, 0.0102], p = 0.9522, fraction positive = 0.517.
- `existing_acoustic_delta`: mean = -0.0007, 95% CI [-0.0091, 0.0078], p = 0.862, fraction positive = 0.512.
- `rich_audio_delta`: mean = -0.0009, 95% CI [-0.0066, 0.0052], p = 0.7754, fraction positive = 0.525.
- `rich_minus_original_delta`: mean = -0.0012, 95% CI [-0.0083, 0.0061], p = 0.7509, fraction positive = 0.507.

Interpretation: the key line is `rich_audio_delta`. If it stays positive, GPT-2 still adds information after stronger waveform/spectral/prosodic controls. If `rich_minus_original_delta` is negative, the richer controls explain part of the original increment.

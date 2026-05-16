from __future__ import annotations
import os

print("Current semantic-fMRI cleaning settings")
print("--------------------------------------")
print("SEMANTIC_FMRI_CONFOUND_STRATEGY =", os.environ.get("SEMANTIC_FMRI_CONFOUND_STRATEGY", "motion24"))
print("SEMANTIC_FMRI_HIGH_PASS         =", os.environ.get("SEMANTIC_FMRI_HIGH_PASS", "none"))
print()
print("Recommended sensitivity ladder:")
print("1) motion24, no high-pass")
print("2) motion6, no high-pass")
print("3) motion24_wmcsf, no high-pass")
print("4) acompcor6, high-pass with fMRIPrep cosine regressors")
print("5) full, high-pass/conservative sensitivity")

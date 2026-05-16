from __future__ import annotations
import os

print("Current semantic-fMRI timing settings")
print("-------------------------------------")
print("SEMANTIC_FMRI_TIMING      =", os.environ.get("SEMANTIC_FMRI_TIMING", "gentle"))
print("SEMANTIC_FMRI_GENTLE_DIR  =", os.environ.get("SEMANTIC_FMRI_GENTLE_DIR", "<path-to-narratives>/stimuli/gentle"))
print("SEMANTIC_FMRI_VERBOSE_TIMING =", os.environ.get("SEMANTIC_FMRI_VERBOSE_TIMING", "0"))
print()
print("Recommended:")
print('$env:SEMANTIC_FMRI_TIMING="gentle"')
print('$env:SEMANTIC_FMRI_GENTLE_DIR="<path-to-narratives>/stimuli/gentle"')
print('$env:SEMANTIC_FMRI_CONFOUND_STRATEGY="motion24"')
print('$env:SEMANTIC_FMRI_HIGH_PASS="none"')


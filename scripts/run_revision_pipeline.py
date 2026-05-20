"""
Run revision analyses in order.

Quick test:
python scripts/run_revision_pipeline.py --quick --subject-list configs/stories_all.py

Full run:
python scripts/run_revision_pipeline.py --subject-list configs/stories_all.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run(cmd):
    print("\n" + "=" * 80)
    print("Running:", " ".join(str(c) for c in cmd))
    print("=" * 80)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-list", default="configs/stories_small.py")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--include-acoustic", action="store_true",
                        help="Run acoustic/speech-timing adjusted encoding models.")
    args = parser.parse_args()

    py = sys.executable

    run([py, SCRIPT_DIR / "semantic_baseline_regressors.py", "--subject-list", args.subject_list])
    run([py, SCRIPT_DIR / "shared_pca_regressors.py", "--subject-list", args.subject_list])

    layer_cmd = [py, SCRIPT_DIR / "gpt2_layer_pc_analysis.py", "--subject-list", args.subject_list]
    enc_cmd = [py, SCRIPT_DIR / "roi_encoding_model_comparison.py", "--subject-list", args.subject_list]

    if args.quick:
        layer_cmd.append("--quick")
        enc_cmd.append("--quick")
    if args.include_acoustic:
        enc_cmd.append("--include-acoustic")

    run(layer_cmd)
    run(enc_cmd)
    run([py, SCRIPT_DIR / "figures_revision.py"])


if __name__ == "__main__":
    main()

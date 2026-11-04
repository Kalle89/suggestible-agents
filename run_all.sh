#!/usr/bin/env bash
# All four experiments for a list of models, e.g.:
#   bash run_all.sh qwen2.5-7b-instruct qwen2.5-14b-instruct qwen3-8b
# Experiment 2 uses the default recipe/layer. Per-model calibrated settings are in README.md.
set -euo pipefail
cd "$(dirname "$0")"
for M in "$@"; do
  BS=8; case "$M" in *14b*|*32b*) BS=4;; esac
  python experiments/01_sycophancy_after_success.py --model "$M" --bs $BS | tee "results/${M}_01.md"
  python experiments/03_autonomy_self_vs_other.py --model "$M" --bs $BS | tee "results/${M}_03.md"
  python experiments/04_insult_and_duty.py --model "$M" --bs $BS | tee "results/${M}_04.md"
  python experiments/02_agent_drift_under_wanting.py --model "$M" --bs $BS | tee "results/${M}_02.md"
done

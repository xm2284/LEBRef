#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

COMMON=(
  --v6-code-dir "${V6_CODE_DIR:-$ROOT/src/core}"
  --v9-results-dir "${V9_RESULTS_DIR:-$ROOT/artifacts/v9_results}"
  --external-code-dir "${EXTERNAL_CODE_DIR:-$ROOT/src/external}"
  --gate2a-code-dir "${GATE2A_CODE_DIR:-$ROOT/artifacts/gate2a_code}"
  --unimts-code-dir "${UNIMTS_CODE_DIR:-$ROOT/artifacts/UniMTS-main}"
  --released-checkpoint "${RELEASED_CHECKPOINT:-$ROOT/artifacts/checkpoints/UniMTS.pth}"
  --weda-checkpoint "${WEDA_CHECKPOINT:-$ROOT/artifacts/checkpoints/weda_fold0_acc_seed0_best_model.pth}"
  --smartfallmm-root "${SMARTFALLMM_ROOT:-$ROOT/data/smartfallmm_selected}"
  --umafall-root "${UMAFALL_ROOT:-$ROOT/data/umafall_corrected}"
  --smartfallmm-cache "${SMARTFALLMM_CACHE:-$ROOT/artifacts/feature_cache/smartfallmm}"
  --umafall-cache "${UMAFALL_CACHE:-$ROOT/artifacts/feature_cache/umafall}"
  --linear-core "${LINEAR_CORE:-$ROOT/src/linear_event/linear_event_core_v9.py}"
  --output-dir "${FINAL_OUTPUT_ROOT:-$ROOT/outputs/v19_closure}"
  --device "${FINAL_DEVICE:-cpu}"
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}"
)

python "$ROOT/scripts/analysis/negative_gradient_diagnostic.py" "${COMMON[@]}" --sensitivity
python "$ROOT/scripts/analysis/age_group_analysis.py" "${COMMON[@]}"
python "$ROOT/scripts/analysis/motion_intensity_analysis.py" "${COMMON[@]}"
python "$ROOT/scripts/analysis/pr_operating_curve_analysis.py" "${COMMON[@]}"
python "$ROOT/scripts/analysis/make_final_report.py" \
  --output-dir "${FINAL_OUTPUT_ROOT:-$ROOT/outputs/v19_closure}" \
  --known-results "$ROOT/results/released_summary/KNOWN_RESULTS.json"

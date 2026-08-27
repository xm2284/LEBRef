# LEBRef

This repository is a curated reproducibility package for:

**LEBRef: Linear Event-Bag Refinement for Continuous Wrist-Worn Fall Detection**

LEBRef is a supervision-refinement method for continuous wrist-worn fall
monitoring. It reorganizes downstream supervision into event bags and
subject-balanced background occupancy so that training better matches the
connected alarm segments used during deployment. The repository provides code,
locked evaluation protocols, and released summary results for the fixed
subject-independent evaluation on SmartFallMM and UMAFall.

## What is included

- Core evaluation/protocol code from the locked V6 pipeline.
- Main V9 linear-event training code for Probe / Event-MIL / LEBRef.
- V19 no-retraining diagnostic scripts for negative-bag gradients, age-group analysis, motion-intensity analysis, and alarm-segment PR / Recall-Duty curves.
- Small released summary results under `results/released_summary/`.
- Audit notes under `docs/audit/`.
- Manuscript-to-repository mapping notes under `docs/MANUSCRIPT_MAPPING.md`.

## What is not included

The repository intentionally does not commit raw datasets, feature caches, trained heads, or checkpoints. These files are large and/or license-controlled. See `docs/DATA_AND_ARTIFACTS.md`.

## Repository layout

```text
src/
  core/          V6 metric, split, feature-cache, and alarm protocol code
  external/      SmartFallMM/UMAFall preprocessing and locked external protocol
  linear_event/  V9 linear event head and losses
scripts/
  reproduce/     main V9 training entry point
  analysis/      V19 no-retraining diagnostics
configs/         fold locks and protocol configs
results/
  released_summary/
docs/
  audit/
  protocols/
tests/
```

## Quick smoke test

```bash
pip install -r requirements.txt
python tests/test_synthetic.py
python -m pytest tests/ -q
```

The smoke tests check local diagnostic functions and import/path contracts only.
They do not require datasets or checkpoints.

## Minimal reproducibility

The small released summaries in `results/released_summary/` allow readers to inspect the locked numerical results without downloading private datasets or model artifacts.
For the Pervasive and Mobile Computing submission layout, see
`docs/MANUSCRIPT_MAPPING.md`.

## Training reproducibility

Training requires:

1. SmartFallMM and UMAFall data obtained under their original licenses.
2. External frozen-representation artifacts:
   - UniMTS source code and `UniMTS.pth`.
   - The WEDA-derived checkpoint (`weda_fold0_acc_seed0_best_model.pth`).
   - The GATE2A code directory that provides `wedafa_common` (used by the V6
     feature-cache builder). See `docs/DATA_AND_ARTIFACTS.md` for roles,
     acquisition, expected paths, and checksum placeholders.
3. The locked V6 reference outputs (under `artifacts/v6_results` and
   `artifacts/v6_legacy_results`), produced by the locked V6 pipeline or
   downloaded from the artifact host.
4. The fold-lock JSON files under `configs/` (committed).
5. Machine-specific paths configured through `.env`.

Copy `.env.example` to `.env`, edit paths, then run the V9 entry point. Complete
V9 main-experiment commands (after datasets/checkpoints are installed):

```bash
# SmartFallMM
python scripts/reproduce/run_experiment_v9.py \
  --dataset smartfallmm \
  --dataset-root "$SMARTFALLMM_ROOT" \
  --v6-code-dir "$V6_CODE_DIR" \
  --v6-results-dir "$V6_RESULTS_DIR" \
  --external-code-dir "$EXTERNAL_CODE_DIR" \
  --gate2a-code-dir "$GATE2A_CODE_DIR" \
  --unimts-code-dir "$UNIMTS_CODE_DIR" \
  --released-checkpoint "$RELEASED_CHECKPOINT" \
  --weda-checkpoint "$WEDA_CHECKPOINT" \
  --cache-dir "$SMARTFALLMM_CACHE" \
  --output-dir "$V9_RESULTS_DIR/smartfallmm" \
  --device cpu

# UMAFall
python scripts/reproduce/run_experiment_v9.py \
  --dataset umafall \
  --dataset-root "$UMAFALL_ROOT" \
  --v6-code-dir "$V6_CODE_DIR" \
  --v6-results-dir "$V6_RESULTS_DIR" \
  --external-code-dir "$EXTERNAL_CODE_DIR" \
  --gate2a-code-dir "$GATE2A_CODE_DIR" \
  --unimts-code-dir "$UNIMTS_CODE_DIR" \
  --released-checkpoint "$RELEASED_CHECKPOINT" \
  --weda-checkpoint "$WEDA_CHECKPOINT" \
  --cache-dir "$UMAFALL_CACHE" \
  --output-dir "$V9_RESULTS_DIR/umafall" \
  --device cpu
```

The command writes per-fold probe / Event-MIL / LEBRef results and the final
`linear_event_results.json`. Do not retrain on the published result directory:
the runner resumes from existing per-fold results and verifies their
`run_config` before reusing them.

## Final diagnostics

After V9 trained-head results are available, run V19 no-retraining diagnostics:

```bash
bash scripts/reproduce/run_v19_analysis.sh
```

This regenerates the post-hoc diagnostic outputs, not the main training results.

## Status

This repository is the public code and reproducibility companion for the
Pervasive and Mobile Computing manuscript. The code is released under the MIT
License. Before creating a DOI archive, attach any large non-Git artifacts, if
needed, through GitHub Releases, Zenodo, or another controlled artifact host.

# LEBRef

This repository is a curated reproducibility package for:

**LEBRef: Linear Event-Bag Refinement for Wrist-Worn Fall Detection**

LEBRef refines a frozen wrist-motion representation with a capacity-matched linear head. The main comparison covers a linear probe, Event-MIL, and LEBRef under subject-independent five-fold evaluation on SmartFallMM and UMAFall.

## What is included

- Core evaluation/protocol code from the locked V6 pipeline.
- Main V9 linear-event training code for Probe / Event-MIL / LEBRef.
- V19 no-retraining diagnostic scripts for negative-bag gradients, age-group analysis, motion-intensity analysis, and alarm-segment PR / Recall-Duty curves.
- Small released summary results under `results/released_summary/`.
- Audit notes under `docs/audit/`.

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
  figures/       figure scripts to be completed/curated
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
```

The smoke test checks local diagnostic functions only. It does not require datasets or checkpoints.

## Minimal reproducibility

The small released summaries in `results/released_summary/` allow readers to inspect the locked numerical results without downloading private datasets or model artifacts.

## Training reproducibility

Training requires:

1. SmartFallMM and UMAFall data obtained under their original licenses.
2. UniMTS source/checkpoint and the WEDA-derived checkpoint used for frozen feature extraction.
3. The fold-lock JSON files under `configs/`.
4. Machine-specific paths configured through `.env`.

Copy `.env.example` to `.env`, edit paths, and then run the V9 entry point in `scripts/reproduce/run_experiment_v9.py`.

## Final diagnostics

After V9 trained-head results are available, run V19 no-retraining diagnostics:

```bash
bash scripts/reproduce/run_v19_analysis.sh
```

This regenerates the post-hoc diagnostic outputs, not the main training results.

## Status

This folder is a GitHub-ready staging package. Before public release, choose a license and attach large artifacts through GitHub Releases or Hugging Face.

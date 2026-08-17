# Data and artifact policy

This repository does not distribute raw datasets, feature caches, or trained checkpoints.

## Datasets

Use the official dataset sources and licenses for:

- SmartFallMM
- UMAFall

Historical/preliminary datasets such as SisFall, WEDA, and KFall are not part of the current main paper experiments and should not be committed.

## External artifacts

The following files should be provided through GitHub Releases, Hugging Face, or another artifact host, with SHA256 checksums:

- `UniMTS.pth`
- WEDA-derived checkpoint(s) used for frozen feature extraction
- optional V9 trained heads
- optional V9 result directory
- optional frozen feature caches

Recommended local placement after download:

```text
artifacts/
  UniMTS-main/
  gate2a_code/
  checkpoints/
    UniMTS.pth
    weda_fold0_acc_seed0_best_model.pth
  feature_cache/
    smartfallmm/
    umafall/
  v9_results/
```

## Do not commit

- raw datasets
- `.pth`, `.pt`, `.ckpt`, `.joblib`
- `.npy`, `.npz`, `.pkl` feature/probability caches
- local logs
- advisor comments or private manuscript notes

# REPRODUCIBILITY REPAIR REPORT

Repository: `xm2284/LEBRef` (public reproducibility skeleton)
Local working copy: `LEBRef_GitHub_Ready_20260817`
Date: 2026-08-17
Scope: make the public skeleton internally executable and path-consistent **without**
changing any reported experimental numbers, folds, seeds, thresholds, or conclusions.

---

## 1. Files changed

| File | Change | Type |
|---|---|---|
| `src/core/run_experiment.py` | New explicit compatibility alias re-exporting `v6_run_experiment` (no scientific code) | added |
| `scripts/reproduce/run_experiment_v9.py` | `config()`: protocol hash now reads `REPO_ROOT/docs/protocols/PROTOCOL_V9.md` instead of `scripts/reproduce/PROTOCOL_V9.md` | modified |
| `src/core/v6_run_experiment.py` | Repo-root-aware `DATASET_CONFIG`: fold locks → `REPO_ROOT/configs/`, protocol → `REPO_ROOT/docs/protocols/PROTOCOL_V6.md`, legacy results → `REPO_ROOT/artifacts/v6_legacy_results/{dataset}` | modified |
| `.env.example` | Added `V6_RESULTS_DIR=./artifacts/v6_results` (non-secret placeholder) | modified |
| `README.md` | Complete V9 commands for SmartFallMM and UMAFall; V6_RESULTS_DIR note; GATE2A pointer; pytest smoke step | modified |
| `docs/DATA_AND_ARTIFACTS.md` | GATE2A role/acquisition/expected path/checksum placeholder; locked V6 reference outputs section; extended `artifacts/` layout | modified |
| `tests/conftest.py` | Inserts `src/core` into `sys.path` so core tests collect (mirrors runtime `V6_CODE_DIR`) | added |
| `tests/test_import_contracts.py` | Import + path-contract smoke tests (no datasets/checkpoints needed) | added |

No scientific variant, seed, fold lock, threshold, or result value was modified.

---

## 2. Issues: confirmed / fixed / not fixed

### Issue 1 — `run_experiment_v9.py` imports `run_experiment`, but `src/core/` exposes `v6_run_experiment.py`
- **Confirmed**: `python -c "import importlib,sys; sys.path.insert(0,'src/core'); importlib.import_module('run_experiment')"` →
  `ModuleNotFoundError: No module named 'run_experiment'`; importing `v6_run_experiment` succeeds.
- **Fixed**: added explicit minimal alias `src/core/run_experiment.py` (re-exports the single canonical implementation). Both
  `scripts/reproduce/run_experiment_v9.py::load_v6` and `scripts/analysis/common.py::load_runtime` now resolve `run_experiment`
  to the same module object as `v6_run_experiment` (verified `alias.resolve_args is canonical.resolve_args == True`).

### Issue 2 — `scripts/analysis/common.py` inherits the same import problem
- **Confirmed**: same `ModuleNotFoundError` at `load_runtime`.
- **Fixed**: same alias covers this call site; no code change needed in `common.py` (its `del sys.modules` guard already prevents
  accidentally importing the analysis package's own modules).

### Issue 3 — `PROTOCOL_V9.md` path under `scripts/reproduce/` does not exist
- **Confirmed**: `PACKAGE_DIR/PROTOCOL_V9.md` (i.e. `scripts/reproduce/`) → `False`; authoritative file is `docs/protocols/PROTOCOL_V9.md` → `True`.
- **Fixed**: `config()` now hashes `REPO_ROOT/docs/protocols/PROTOCOL_V9.md`.
- **Numerical-safety check**: sha256 of `docs/protocols/PROTOCOL_V9.md` =
  `17c51b3704345dda9257dc59789f01ec49cce6d178efdfb535b8ccbb3ca7b8ef`, which **exactly matches** the `protocol_sha256`
  embedded in both released `results/released_summary/v9_main/*.json`. Resume-config comparison therefore stays valid and no
  result value changes.

### Issue 4 — `src/core/v6_run_experiment.py` relative paths vs. curated layout
- **Confirmed**: `DATASET_CONFIG` pointed at `src/core/configs/`, `src/core/PROTOCOL_V6.md`, `src/core/legacy_results/` — none match
  the curated layout (fold locks live in `configs/`, protocols in `docs/protocols/`, legacy results are not committed).
- **Fixed**: repo-root-aware resolution —
  - fold lock → `REPO_ROOT/configs/{dataset}_fold_lock.json` (committed, exists);
  - protocol → `REPO_ROOT/docs/protocols/PROTOCOL_V6.md` (committed, exists);
  - legacy results → `REPO_ROOT/artifacts/v6_legacy_results/{dataset}` (documented external artifact, still validated by `resolve_args`).
- Locked files themselves were **not** modified.

### Issue 5 — V9 requires `--v6-results-dir`; `.env.example`/README incomplete
- **Confirmed**: `.env.example` had no `V6_RESULTS_DIR`; README had no executable V9 command.
- **Fixed**: `.env.example` adds `V6_RESULTS_DIR=./artifacts/v6_results`; README now contains one complete V9 main-experiment
  command per dataset (SmartFallMM and UMAFall), covering all 11 required flags, plus a resume-verification warning.

### Issue 6 — GATE2A acquisition/placement insufficiently documented
- **Confirmed**: `docs/DATA_AND_ARTIFACTS.md` listed `gate2a_code/` in the layout but gave no role/acquisition/checksum guidance.
- **Fixed**: new "GATE2A code directory" section documents its exact role
  (`legacy_smartfallmm.configure_imports` → `wedafa_common.build_model_from_checkpoint`), acquisition (WEDA gate-2a source,
  author-published tarball), expected path (`artifacts/gate2a_code/` containing `wedafa_common.py`), and an explicit
  checksum placeholder (intentionally empty — no silently-assumed artifact).

### Not fixed / out of scope
- `tests/verify_v6_package.py` and `tests/verify_v19_package.py` target the *old embedded-package* layout (paths relative to
  `tests/`, `embedded_sources/`, etc.). They are not part of the curated layout, are not invoked by any runtime path, and were
  left untouched (they are not run by the smoke suite).
- No data/preprocessing files were touched; datasets remain external.

---

## 3. Validation commands and outputs

Environment: Python 3.14.0, torch 2.11.0+cpu, sklearn 1.8.0, numpy 2.4.4.

### 3.1 `python -m compileall src scripts tests` → `compileall_exit=0`

### 3.2 `python -m pytest tests/ -q` → `13 passed in 14.21s`
Before the fix the suite could not even collect:
`tests\test_v6_core.py:9: ModuleNotFoundError: No module named 'esra_core'`.
After `tests/conftest.py`, all core + import-contract tests pass (no datasets/checkpoints involved).

### 3.3 V6 runner import smoke from the exact default V6_CODE_DIR (`src/core`)
```text
alias import OK -> run_experiment
canonical import OK -> v6_run_experiment
resolve_args identical: True
```

### 3.4 `python scripts/reproduce/run_experiment_v9.py --help` → exit=0
Usage shows all 11 required path flags (`--dataset-root --v6-code-dir --v6-results-dir --external-code-dir --gate2a-code-dir
--unimts-code-dir --released-checkpoint --weda-checkpoint --cache-dir --output-dir --device`).

### 3.5 Analysis entry-point `--help` → all exit=0
`negative_gradient_diagnostic.py`, `age_group_analysis.py`, `motion_intensity_analysis.py`, `pr_operating_curve_analysis.py`,
`make_final_report.py`.

### 3.6 `.env.example` path mapping (every key maps to a documented location; all repo-relative)
```text
SMARTFALLMM_ROOT / UMAFALL_ROOT      -> data/            (DATA_AND_ARTIFACTS: datasets)
UNIMTS_CODE_DIR / GATE2A_CODE_DIR    -> artifacts/       (external artifacts)
RELEASED_CHECKPOINT / WEDA_CHECKPOINT-> artifacts/       (external artifacts)
SMARTFALLMM_CACHE / UMAFALL_CACHE    -> artifacts/       (external artifacts)
V9_RESULTS_DIR / V6_RESULTS_DIR      -> artifacts/       (external artifacts)
V6_CODE_DIR / EXTERNAL_CODE_DIR / LINEAR_CORE -> src/    (committed)
FINAL_OUTPUT_ROOT                    -> outputs/         (gitignored run output)
FINAL_DEVICE / BOOTSTRAP_ITERATIONS  -> non-path values
ALL_OK
```

### 3.7 Sensitive-information grep
Patterns: `/mnt/`, `C:\Users`, `/home/`, `lenovo`, `token`, `api[_-]?key`, `password`, `secret`, `BEGIN (RSA|OPENSSH|PRIVATE)`.
Result: only match is `.gitignore:45: *secret*` (the ignore rule itself). No private absolute paths, credentials, tokens, or
local usernames found.

### 3.8 V6 `resolve_args` path-contract smoke
```text
fold_lock -> <repo>/configs/smartfallmm_fold_lock.json   (committed, exists)
protocol  -> <repo>/docs/protocols/PROTOCOL_V6.md        (committed, exists)
legacy    -> <repo>/artifacts/v6_legacy_results/smartfallmm (documented external artifact)
```
`resolve_args` stops deterministically on the first missing external artifact (datasets/checkpoints), i.e. it now fails only on
the documented artifact set, never on repository-internal paths.

---

## 4. Still required from the author (external artifacts)

None of these may be committed to Git; they must be hosted (GitHub Releases / Hugging Face / Zenodo) with SHA256:

1. SmartFallMM and UMAFall raw datasets (official sources/licenses).
2. UniMTS source (`artifacts/UniMTS-main/`) and `UniMTS.pth` (`artifacts/checkpoints/UniMTS.pth`).
3. WEDA-derived checkpoint `weda_fold0_acc_seed0_best_model.pth` (15 WEDA fold/seed checkpoints in the original layout).
4. GATE2A code directory with `wedafa_common.py` (+ imported modules) — role/acquisition documented in `DATA_AND_ARTIFACTS.md`.
5. Locked V6 reference outputs `artifacts/v6_results/` (esra_ablation + locked_alarm_duty_audit).
6. Locked V6 legacy records `artifacts/v6_legacy_results/{smartfallmm,umafall}/` (fivefold_results.json + fold*/seed*/result.json + best_head.pth).
7. Optional: frozen feature caches, published V9 result directory (`artifacts/v9_results/`) and trained heads.

Items 5–6 are **mandatory** for both V9 training and most V19 diagnostics, because `resolve_args` validates
`fivefold_results.json` even when only `load_runtime` is used.

---

## 5. Fresh-user readiness

- **Level 1 — results/diagnostics**
  - Inspect-only: **reachable now**. `results/released_summary/` (KNOWN_RESULTS + v9_main + v19_closure) requires no artifacts.
  - V19 no-retraining diagnostics (`bash scripts/reproduce/run_v19_analysis.sh`): **import/path layer fixed and verified**, but
    3 of 4 diagnostics call `load_runtime` and therefore need the external artifact set (datasets + checkpoints + gate2a +
    feature caches + v6 legacy results) plus a downloaded `v9_results`. `age_group_analysis` and `make_final_report` work from
    `v9_results` + released summaries alone.
- **Level 2 — training (V9 main experiment)**: **import/path layer fixed and verified** (`--help` OK, V6 runner imports from the
  default `V6_CODE_DIR`, protocol hash matches released results, all documented paths resolve). Full training additionally
  requires installing all external artifacts in §4 (datasets, checkpoints, gate2a, v6_results, v6_legacy_results) and then
  running the README commands; no training was launched during this repair.

**Summary**: the repository skeleton is now internally executable and path-consistent with zero scientific drift. Reaching
Level 1 (diagnostics) and Level 2 (training) now depends only on the author publishing the external artifacts in §4.

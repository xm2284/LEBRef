"""Import and path-contract smoke tests.

These tests validate that the curated repository's runtime imports and default
paths are consistent. They require neither datasets nor checkpoints.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_CORE = REPO_ROOT / "src" / "core"
REQUIRED_ENV_KEYS = (
    "SMARTFALLMM_ROOT",
    "UMAFALL_ROOT",
    "UNIMTS_CODE_DIR",
    "GATE2A_CODE_DIR",
    "RELEASED_CHECKPOINT",
    "WEDA_CHECKPOINT",
    "SMARTFALLMM_CACHE",
    "UMAFALL_CACHE",
    "V9_RESULTS_DIR",
    "V6_RESULTS_DIR",
    "V6_CODE_DIR",
    "EXTERNAL_CODE_DIR",
    "LINEAR_CORE",
    "FINAL_OUTPUT_ROOT",
    "FINAL_DEVICE",
    "BOOTSTRAP_ITERATIONS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_v6_runner(module_name: str):
    if str(SRC_CORE) not in sys.path:
        sys.path.insert(0, str(SRC_CORE))
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def test_v6_runner_imports_from_default_v6_code_dir() -> None:
    # Default V6_CODE_DIR is REPO_ROOT/src/core, which exposes v6_run_experiment.py.
    runner = _load_v6_runner("v6_run_experiment")
    assert hasattr(runner, "resolve_args")
    assert hasattr(runner, "load_runtime")
    assert hasattr(runner, "split_subjects")
    assert hasattr(runner, "run_esra")


def test_run_experiment_compat_alias_is_identical() -> None:
    # scripts/reproduce/run_experiment_v9.py and scripts/analysis/common.py import
    # "run_experiment"; src/core/run_experiment.py is the explicit alias.
    canonical = _load_v6_runner("v6_run_experiment")
    alias = _load_v6_runner("run_experiment")
    assert alias.resolve_args is canonical.resolve_args
    assert alias.load_runtime is canonical.load_runtime
    assert alias.DATASET_CONFIG == canonical.DATASET_CONFIG


def test_v6_paths_are_repo_root_aware() -> None:
    runner = _load_v6_runner("v6_run_experiment")
    for dataset in ("smartfallmm", "umafall"):
        fold_lock = runner.DATASET_CONFIG[dataset]["fold_lock"]
        assert fold_lock.resolve() == (REPO_ROOT / "configs" / f"{dataset}_fold_lock.json").resolve()
        assert fold_lock.is_file(), f"missing committed fold lock: {fold_lock}"
        legacy = runner.DATASET_CONFIG[dataset]["legacy_results"]
        assert legacy.resolve() == (REPO_ROOT / "artifacts" / "v6_legacy_results" / dataset).resolve()
    protocol = REPO_ROOT / "docs" / "protocols" / "PROTOCOL_V6.md"
    assert protocol.is_file(), "missing locked V6 protocol"


def test_v9_protocol_sha256_matches_released_results() -> None:
    # The deterministic V9 protocol path (REPO_ROOT/docs/protocols/PROTOCOL_V9.md)
    # must hash to the same value embedded in the released results, otherwise the
    # resume run_config check would break and the numbers would change.
    protocol = REPO_ROOT / "docs" / "protocols" / "PROTOCOL_V9.md"
    assert protocol.is_file()
    published = json.loads(
        (
            REPO_ROOT
            / "results"
            / "released_summary"
            / "v9_main"
            / "smartfallmm_linear_event_results.json"
        ).read_text(encoding="utf-8")
    )
    published_sha = published["run_config"]["protocol_sha256"]
    assert _sha256(protocol) == published_sha


def test_v9_config_uses_deterministic_protocol_path() -> None:
    scripts_reproduce = str((REPO_ROOT / "scripts" / "reproduce").resolve())
    if scripts_reproduce not in sys.path:
        sys.path.insert(0, scripts_reproduce)
    import run_experiment_v9 as v9

    cfg = v9.config(
        SimpleNamespace(
            dataset="smartfallmm",
            epochs=15,
            steps_per_epoch=16,
            learning_rate=3e-4,
            weight_decay=1e-4,
            positive_bags_per_step=8,
            negative_bags_per_step=8,
            subjects_per_step=8,
            background_windows_per_subject=128,
        )
    )
    assert cfg["protocol_sha256"] == _sha256(REPO_ROOT / "docs" / "protocols" / "PROTOCOL_V9.md")
    assert cfg["dataset"] == "smartfallmm"
    assert cfg["seeds"] == [20260724, 20260725, 20260726]


def test_env_example_exposes_all_runtime_keys() -> None:
    env = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    missing = [key for key in REQUIRED_ENV_KEYS if f"{key}=" not in env]
    assert not missing, f".env.example missing keys: {missing}"

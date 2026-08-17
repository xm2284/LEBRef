from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import torch

PRIMARY_SEED = 20260724
SEEDS = (20260724, 20260725, 20260726)
METHOD_DIRS = {
    "probe": "linear_probe",
    "event_mil": "linear_event_mil",
    "lebref": "linear_event_duty",
}
REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _candidate_dataset_roots(base: Path, dataset: str) -> list[Path]:
    roots = [base / dataset, base]
    return [p for i, p in enumerate(roots) if p.exists() and p not in roots[:i]]


def _matching_json_dataset(path: Path, dataset: str) -> bool:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    candidates = [
        d.get("dataset"),
        d.get("run_config", {}).get("dataset") if isinstance(d.get("run_config"), dict) else None,
    ]
    return dataset in candidates or all(x is None for x in candidates)


def find_fold_result(v9_results_dir: Path, dataset: str, fold: int, method: str, seed: int = PRIMARY_SEED) -> Path:
    method_dir = METHOD_DIRS[method]
    rels: list[Path] = []
    if method == "probe":
        rels.append(Path(f"fold{fold}") / method_dir / "result.json")
    else:
        rels.append(Path(f"fold{fold}") / method_dir / f"seed{seed}" / "result.json")
    for root in _candidate_dataset_roots(v9_results_dir, dataset):
        for rel in rels:
            p = root / rel
            if p.is_file() and _matching_json_dataset(p, dataset):
                return p
    pattern = f"**/fold{fold}/{method_dir}/result.json" if method == "probe" else f"**/fold{fold}/{method_dir}/seed{seed}/result.json"
    matches = [p for p in v9_results_dir.glob(pattern) if _matching_json_dataset(p, dataset)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No {method} result found for {dataset} fold={fold} seed={seed} under {v9_results_dir}")
    raise RuntimeError(f"Ambiguous {method} result for {dataset} fold={fold} seed={seed}: {matches}")


def find_head_path(v9_results_dir: Path, dataset: str, fold: int, method: str, seed: int = PRIMARY_SEED) -> Path:
    if method == "probe":
        result = find_fold_result(v9_results_dir, dataset, fold, method, seed)
        p = result.with_name("model.joblib")
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    result = find_fold_result(v9_results_dir, dataset, fold, method, seed)
    p = result.with_name("selected_head.pth")
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_result(v9_results_dir: Path, dataset: str, fold: int, method: str, seed: int = PRIMARY_SEED) -> dict[str, Any]:
    return json.loads(find_fold_result(v9_results_dir, dataset, fold, method, seed).read_text(encoding="utf-8"))


def load_oof_details(v9_results_dir: Path, dataset: str, method: str, seed: int = PRIMARY_SEED) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fold in range(5):
        out.extend(load_result(v9_results_dir, dataset, fold, method, seed)["test_details"])
    return out


def subject_false_alarm_duty(details: Sequence[dict[str, Any]]) -> dict[str, float]:
    accum: dict[str, list[float]] = {}
    for r in details:
        s = str(r["subject"])
        bg = float(r.get("non_event_monitoring_seconds", 0.0) or 0.0)
        fa = float(r.get("event_external_alarm_seconds", r.get("alarm_seconds", 0.0)) or 0.0)
        if s not in accum:
            accum[s] = [0.0, 0.0]
        accum[s][0] += fa
        accum[s][1] += bg
    return {s: (fa / bg if bg > 0 else float("nan")) for s, (fa, bg) in accum.items()}


def subject_fa_per_hour(details: Sequence[dict[str, Any]]) -> dict[str, float]:
    accum: dict[str, list[float]] = {}
    for r in details:
        s = str(r["subject"])
        bg = float(r.get("non_event_monitoring_seconds", 0.0) or 0.0)
        nfa = float(r.get("false_alarms", 0.0) or 0.0)
        if s not in accum:
            accum[s] = [0.0, 0.0]
        accum[s][0] += nfa
        accum[s][1] += bg
    return {s: (nfa / (bg / 3600.0) if bg > 0 else float("nan")) for s, (nfa, bg) in accum.items()}


def paired_bootstrap(values_a: dict[str, float], values_b: dict[str, float], iterations: int, seed: int) -> dict[str, float]:
    common = sorted(set(values_a) & set(values_b))
    a = np.asarray([values_a[s] for s in common], dtype=np.float64)
    b = np.asarray([values_b[s] for s in common], dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    common = [s for s, keep in zip(common, mask) if keep]
    if len(a) == 0:
        raise ValueError("No finite paired subjects")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(iterations, len(diff)))
    bs = diff[idx].mean(axis=1)
    return {
        "n_subjects": len(diff),
        "mean_difference": float(diff.mean()),
        "ci_low": float(np.quantile(bs, 0.025)),
        "ci_high": float(np.quantile(bs, 0.975)),
    }


def bootstrap_mean(values: Sequence[float], iterations: int, seed: int) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(iterations, len(x)))
    bs = x[idx].mean(axis=1)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "ci_low": float(np.quantile(bs, 0.025)),
        "ci_high": float(np.quantile(bs, 0.975)),
    }


def diff_in_group_means_bootstrap(diff: dict[str, float], group: dict[str, str], group_a: str, group_b: str, iterations: int, seed: int) -> dict[str, float]:
    a = np.asarray([diff[s] for s in diff if group.get(s) == group_a and np.isfinite(diff[s])], dtype=float)
    b = np.asarray([diff[s] for s in diff if group.get(s) == group_b and np.isfinite(diff[s])], dtype=float)
    if not len(a) or not len(b):
        raise ValueError("empty group")
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(iterations, len(a)))
    ib = rng.integers(0, len(b), size=(iterations, len(b)))
    bs = a[ia].mean(1) - b[ib].mean(1)
    est = float(a.mean() - b.mean())
    p = float((1 + min((bs <= 0).sum(), (bs >= 0).sum()) * 2) / (iterations + 1))
    return {
        "group_a": group_a,
        "group_b": group_b,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "difference_of_mean_differences": est,
        "ci_low": float(np.quantile(bs, 0.025)),
        "ci_high": float(np.quantile(bs, 0.975)),
        "bootstrap_two_sided_p_approx": min(1.0, p),
    }


def trial_is_fall(trial: Any) -> bool:
    return bool(getattr(trial, "is_fall"))


def load_runtime(args: Any, dataset: str) -> tuple[Any, Any, Any]:
    v6_code_dir = Path(args.v6_code_dir).resolve()
    if str(v6_code_dir) not in sys.path:
        sys.path.insert(0, str(v6_code_dir))
    # Protect against importing this package's scripts/common instead of V6 runner.
    module_name = "run_experiment"
    if module_name in sys.modules:
        del sys.modules[module_name]
    v6_runner = importlib.import_module(module_name)
    v6_core = importlib.import_module("esra_core")
    dataset_root = Path(args.smartfallmm_root if dataset == "smartfallmm" else args.umafall_root).resolve()
    cache_dir = Path(args.smartfallmm_cache if dataset == "smartfallmm" else args.umafall_cache).resolve()
    output_dir = Path(args.output_dir).resolve() / "_runtime_tmp" / dataset
    v6_args = SimpleNamespace(
        dataset=dataset,
        mode="esra",
        dataset_root=dataset_root,
        external_code_dir=Path(args.external_code_dir).resolve(),
        gate2a_code_dir=Path(args.gate2a_code_dir).resolve(),
        unimts_code_dir=Path(args.unimts_code_dir).resolve(),
        released_checkpoint=Path(args.released_checkpoint).resolve(),
        weda_checkpoint=Path(args.weda_checkpoint).resolve(),
        cache_dir=cache_dir,
        output_dir=output_dir,
        device=getattr(args, "device", "cpu"),
        feature_batch_size=getattr(args, "feature_batch_size", 1024),
        bootstrap_iterations=getattr(args, "bootstrap_iterations", 10000),
        epochs=15,
        patience=15,
        steps_per_epoch=16,
        learning_rate=3e-4,
        weight_decay=1e-4,
        positive_bags_per_step=8,
        negative_bags_per_step=8,
        subjects_per_step=8,
        background_windows_per_subject=128,
        temporal_bags_per_step=1,
        variants=["event_mil"],
    )
    v6_args = v6_runner.resolve_args(v6_args)
    runtime = v6_runner.load_runtime(v6_args)
    return v6_runner, v6_core, runtime


def load_linear_core(linear_core_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lebref_linear_core_v9", str(linear_core_path))
    if spec is None or spec.loader is None:
        raise ImportError(linear_core_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_head(linear_core: Any, v9_results_dir: Path, dataset: str, fold: int, method: str, seed: int, device: str = "cpu"):
    if method == "probe":
        model = joblib.load(find_head_path(v9_results_dir, dataset, fold, method, seed))
        head = linear_core.head_from_pipeline(model)
        return head.to(device)
    p = find_head_path(v9_results_dir, dataset, fold, method, seed)
    payload = torch.load(p, map_location="cpu", weights_only=False)
    state = payload.get("head_state_dict", payload)
    dim = int(state["feature_mean"].numel())
    head = linear_core.LinearEventHead(np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32))
    head.load_state_dict(state, strict=True)
    return head.to(device)


def method_threshold(v9_results_dir: Path, dataset: str, fold: int, method: str, seed: int = PRIMARY_SEED) -> float:
    r = load_result(v9_results_dir, dataset, fold, method, seed)
    if method == "probe":
        sc = r.get("selected_candidate", {})
        for key in ("selected_threshold", "threshold"):
            if key in sc:
                return float(sc[key])
        return float(r["test"]["threshold"])
    sc = r.get("selected_candidate", {})
    if "selected_threshold" in sc:
        return float(sc["selected_threshold"])
    tr = sc.get("training", {}) if isinstance(sc, dict) else {}
    if "selected_threshold" in tr:
        return float(tr["selected_threshold"])
    return float(r["test"]["threshold"])


def add_runtime_args(parser) -> None:
    parser.add_argument("--v6-code-dir", default=os.environ.get("V6_CODE_DIR", str(REPO_ROOT / "src" / "core")))
    parser.add_argument("--v9-results-dir", default=os.environ.get("V9_RESULTS_DIR", str(REPO_ROOT / "artifacts" / "v9_results")))
    parser.add_argument("--external-code-dir", default=os.environ.get("EXTERNAL_CODE_DIR", str(REPO_ROOT / "src" / "external")))
    parser.add_argument("--gate2a-code-dir", default=os.environ.get("GATE2A_CODE_DIR", str(REPO_ROOT / "artifacts" / "gate2a_code")))
    parser.add_argument("--unimts-code-dir", default=os.environ.get("UNIMTS_CODE_DIR", str(REPO_ROOT / "artifacts" / "UniMTS-main")))
    parser.add_argument("--released-checkpoint", default=os.environ.get("RELEASED_CHECKPOINT", str(REPO_ROOT / "artifacts" / "checkpoints" / "UniMTS.pth")))
    parser.add_argument("--weda-checkpoint", default=os.environ.get("WEDA_CHECKPOINT", str(REPO_ROOT / "artifacts" / "checkpoints" / "weda_fold0_acc_seed0_best_model.pth")))
    parser.add_argument("--smartfallmm-root", default=os.environ.get("SMARTFALLMM_ROOT", str(REPO_ROOT / "data" / "smartfallmm_selected")))
    parser.add_argument("--umafall-root", default=os.environ.get("UMAFALL_ROOT", str(REPO_ROOT / "data" / "umafall_corrected")))
    parser.add_argument("--smartfallmm-cache", default=os.environ.get("SMARTFALLMM_CACHE", str(REPO_ROOT / "artifacts" / "feature_cache" / "smartfallmm")))
    parser.add_argument("--umafall-cache", default=os.environ.get("UMAFALL_CACHE", str(REPO_ROOT / "artifacts" / "feature_cache" / "umafall")))
    parser.add_argument("--linear-core", default=os.environ.get("LINEAR_CORE", str(REPO_ROOT / "src" / "linear_event" / "linear_event_core_v9.py")))
    parser.add_argument("--output-dir", default=os.environ.get("FINAL_OUTPUT_ROOT", str(REPO_ROOT / "outputs" / "v19_closure")))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=os.environ.get("FINAL_DEVICE", "cpu"))
    parser.add_argument("--feature-batch-size", type=int, default=int(os.environ.get("FEATURE_BATCH_SIZE", "1024")))
    parser.add_argument("--bootstrap-iterations", type=int, default=int(os.environ.get("BOOTSTRAP_ITERATIONS", "10000")))


def validate_runtime_paths(args: Any, need_v9: bool = True) -> None:
    paths = [
        args.v6_code_dir,
        args.external_code_dir,
        args.gate2a_code_dir,
        args.unimts_code_dir,
        args.released_checkpoint,
        args.weda_checkpoint,
        args.smartfallmm_root,
        args.umafall_root,
        args.smartfallmm_cache,
        args.umafall_cache,
        args.linear_core,
    ]
    if need_v9:
        paths.append(args.v9_results_dir)
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))

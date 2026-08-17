from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from common import (
    SEEDS, add_runtime_args, atomic_json, bootstrap_mean, load_result, load_runtime,
    subject_false_alarm_duty, validate_runtime_paths, write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-retraining subject-level motion-intensity heterogeneity analysis.")
    add_runtime_args(p)
    p.add_argument("--datasets", nargs="+", default=["smartfallmm", "umafall"], choices=["smartfallmm", "umafall"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return p.parse_args()


def xyz_from_signal(signal: Any) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"trial.signal must be 2D, got {x.shape}")
    if x.shape[1] >= 3:
        xyz = x[:, :3]
    elif x.shape[0] >= 3:
        xyz = x[:3].T
    else:
        raise ValueError(f"cannot find 3 acceleration axes in shape {x.shape}")
    return xyz[np.all(np.isfinite(xyz), axis=1)]


def intensity_metrics(signal: Any, sample_rate_hz: float = 50.0) -> dict[str, float]:
    xyz = xyz_from_signal(signal)
    if len(xyz) < 3:
        return {"dynamic_vm_rms": float("nan"), "jerk_rms": float("nan"), "vm_iqr": float("nan")}
    vm = np.sqrt(np.sum(xyz * xyz, axis=1))
    centered = vm - np.median(vm)
    dynamic_vm_rms = float(np.sqrt(np.mean(centered * centered)))
    jerk = np.diff(xyz, axis=0) * sample_rate_hz
    jerk_rms = float(np.sqrt(np.mean(np.sum(jerk * jerk, axis=1)))) if len(jerk) else float("nan")
    vm_iqr = float(np.quantile(vm, 0.75) - np.quantile(vm, 0.25))
    return {"dynamic_vm_rms": dynamic_vm_rms, "jerk_rms": jerk_rms, "vm_iqr": vm_iqr}


def collect_subject_intensity(args: argparse.Namespace, dataset: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    v6_runner, _, runtime = load_runtime(args, dataset)
    trial_rows: list[dict[str, Any]] = []
    by_subject: dict[str, list[float]] = {}
    for fold in range(5):
        _, _, test_subjects = v6_runner.split_subjects(runtime.folds, fold)
        test_trials, _, _ = runtime.legacy.subset(runtime.trials, runtime.offsets, test_subjects)
        probe_record = load_result(Path(args.v9_results_dir), dataset, fold, "probe", SEEDS[0])
        details = probe_record["test_details"]
        if len(details) != len(test_trials):
            raise RuntimeError(f"trial/detail count mismatch {dataset} fold {fold}: {len(test_trials)} vs {len(details)}")
        for trial, detail in zip(test_trials, details):
            if str(trial.subject) != str(detail["subject"]):
                raise RuntimeError(f"trial/detail subject mismatch {dataset} fold {fold}: {trial.subject} vs {detail['subject']}")
            if bool(trial.is_fall):
                continue
            m = intensity_metrics(trial.signal, 50.0)
            value = m["dynamic_vm_rms"]
            trial_rows.append({"dataset": dataset, "fold": fold, "subject": str(trial.subject), **m})
            if np.isfinite(value):
                by_subject.setdefault(str(trial.subject), []).append(value)
    subject_intensity = {s: float(np.median(v)) for s, v in by_subject.items() if v}
    return subject_intensity, trial_rows


def assign_terciles(values: dict[str, float]) -> tuple[dict[str, str], float, float]:
    x = np.asarray(list(values.values()), dtype=float)
    q1, q2 = np.quantile(x, [1/3, 2/3])
    groups = {}
    for s, v in values.items():
        groups[s] = "low" if v <= q1 else ("middle" if v <= q2 else "high")
    return groups, float(q1), float(q2)


def details_for_method(args: argparse.Namespace, dataset: str, method: str, seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fold in range(5):
        out.extend(load_result(Path(args.v9_results_dir), dataset, fold, method, seed)["test_details"])
    return out


def main() -> None:
    args = parse_args()
    validate_runtime_paths(args)
    out_root = Path(args.output_dir) / "final_new_analyses" / "motion_intensity"
    out_root.mkdir(parents=True, exist_ok=True)
    all_trial_rows: list[dict[str, Any]] = []
    all_subject_rows: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        intensity, trial_rows = collect_subject_intensity(args, dataset)
        all_trial_rows.extend(trial_rows)
        groups, q1, q2 = assign_terciles(intensity)
        for seed in args.seeds:
            probe = subject_false_alarm_duty(details_for_method(args, dataset, "probe", seed))
            leb = subject_false_alarm_duty(details_for_method(args, dataset, "lebref", seed))
            common = sorted(set(probe) & set(leb) & set(intensity))
            deltas = {s: leb[s] - probe[s] for s in common}
            rho, p = spearmanr([intensity[s] for s in common], [deltas[s] for s in common])
            corr_rows.append({"dataset": dataset, "seed": seed, "n_subjects": len(common), "spearman_rho_intensity_vs_delta_duty": float(rho), "spearman_p": float(p)})
            for s in common:
                all_subject_rows.append({
                    "dataset": dataset, "seed": seed, "subject": s, "motion_intensity_dynamic_vm_rms": intensity[s],
                    "intensity_tercile": groups[s], "probe_duty": probe[s], "lebref_duty": leb[s], "delta_duty": deltas[s],
                    "tercile_q1": q1, "tercile_q2": q2,
                })
            for g in ("low", "middle", "high"):
                members = [s for s in common if groups[s] == g]
                dstat = bootstrap_mean([deltas[s] for s in members], args.bootstrap_iterations, 20260820 + seed % 1000 + {"low":0,"middle":10,"high":20}[g])
                pstat = bootstrap_mean([probe[s] for s in members], args.bootstrap_iterations, 20260821 + seed % 1000 + {"low":0,"middle":10,"high":20}[g])
                lstat = bootstrap_mean([leb[s] for s in members], args.bootstrap_iterations, 20260822 + seed % 1000 + {"low":0,"middle":10,"high":20}[g])
                all_group_rows.append({
                    "dataset": dataset, "seed": seed, "intensity_tercile": g, "n_subjects": len(members),
                    "probe_mean_duty": pstat["mean"], "lebref_mean_duty": lstat["mean"],
                    "mean_delta_duty": dstat["mean"], "delta_ci_low": dstat["ci_low"], "delta_ci_high": dstat["ci_high"],
                    "q1": q1, "q2": q2,
                })
    write_csv(out_root / "trial_motion_intensity.csv", all_trial_rows)
    write_csv(out_root / "subject_motion_intensity_and_duty.csv", all_subject_rows)
    write_csv(out_root / "motion_intensity_tercile_summary.csv", all_group_rows)
    write_csv(out_root / "motion_intensity_correlation.csv", corr_rows)
    atomic_json(out_root / "motion_intensity_manifest.json", {
        "state": "complete", "analysis": "subject-level motion-intensity heterogeneity, no retraining",
        "primary_intensity_metric": "median across non-fall trials of RMS deviation of acceleration vector magnitude from its trial median",
        "terciles": "dataset-specific subject-level terciles computed before looking at alarm outcomes",
        "caution": "Exploratory heterogeneity analysis; movement intensity is an accelerometer-derived proxy, not a clinical activity-intensity label.",
    })
    print("MOTION_INTENSITY_ANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import esra_core as core


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DATASET_CONFIG = {
    "smartfallmm": {
        "legacy_module": "legacy_smartfallmm",
        "fold_lock": REPO_ROOT / "configs" / "smartfallmm_fold_lock.json",
        "legacy_results": REPO_ROOT / "artifacts" / "v6_legacy_results" / "smartfallmm",
    },
    "umafall": {
        "legacy_module": "legacy_umafall",
        "fold_lock": REPO_ROOT / "configs" / "umafall_fold_lock.json",
        "legacy_results": REPO_ROOT / "artifacts" / "v6_legacy_results" / "umafall",
    },
}
VARIANTS = ("event_mil", "event_duty", "event_duty_cvar", "esra_full")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked duration audit and ESRA-Head subject-independent ablation."
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_CONFIG), required=True)
    parser.add_argument("--mode", choices=("audit", "esra", "all"), default="audit")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--external-code-dir", type=Path, required=True)
    parser.add_argument("--gate2a-code-dir", type=Path, required=True)
    parser.add_argument("--unimts-code-dir", type=Path, required=True)
    parser.add_argument("--released-checkpoint", type=Path, required=True)
    parser.add_argument("--weda-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-bags-per-step", type=int, default=8)
    parser.add_argument("--negative-bags-per-step", type=int, default=8)
    parser.add_argument("--subjects-per-step", type=int, default=8)
    parser.add_argument("--background-windows-per-subject", type=int, default=128)
    parser.add_argument("--temporal-bags-per-step", type=int, default=8)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "dataset_root",
        "external_code_dir",
        "gate2a_code_dir",
        "unimts_code_dir",
        "released_checkpoint",
        "weda_checkpoint",
        "cache_dir",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.fold_lock = DATASET_CONFIG[args.dataset]["fold_lock"].resolve()
    args.legacy_results = DATASET_CONFIG[args.dataset]["legacy_results"].resolve()
    args.protocol = (REPO_ROOT / "docs" / "protocols" / "PROTOCOL_V6.md").resolve()
    args.protocol_sha256 = file_sha256(args.protocol)
    if args.bootstrap_iterations < 1000:
        raise ValueError("bootstrap iterations must be at least 1000")
    for name in ("epochs", "patience", "steps_per_epoch"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for path in (
        args.dataset_root,
        args.external_code_dir,
        args.gate2a_code_dir,
        args.unimts_code_dir,
        args.cache_dir.parent,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (
        args.released_checkpoint,
        args.weda_checkpoint,
        args.fold_lock,
        args.protocol,
        args.legacy_results / "fivefold_results.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def training_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol_sha256": args.protocol_sha256,
        "epochs": args.epochs,
        "patience": args.patience,
        "steps_per_epoch": args.steps_per_epoch,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "positive_bags_per_step": args.positive_bags_per_step,
        "negative_bags_per_step": args.negative_bags_per_step,
        "subjects_per_step": args.subjects_per_step,
        "background_windows_per_subject": args.background_windows_per_subject,
        "temporal_bags_per_step": args.temporal_bags_per_step,
    }


def load_runtime(args: argparse.Namespace) -> SimpleNamespace:
    legacy = importlib.import_module(DATASET_CONFIG[args.dataset]["legacy_module"])
    external, weda_common = legacy.configure_imports(args)
    trials, dataset_audit = external.load_external_trials(args.dataset, args.dataset_root)
    offsets, total_windows = legacy.attach_offsets(external, trials)
    folds, fold_payload = legacy.load_fold_lock(
        args.fold_lock, {str(trial.subject) for trial in trials}
    )
    progress_path = args.output_dir / "progress.json"
    features, zero_scores, cache_manifest = legacy.create_or_load_cache(
        args,
        external,
        weda_common,
        trials,
        offsets,
        total_windows,
        dataset_audit["model_input_sha256"],
        progress_path,
    )
    initial_state = torch.load(
        legacy.cache_paths(args.cache_dir)["head"], map_location="cpu", weights_only=False
    )
    return SimpleNamespace(
        legacy=legacy,
        external=external,
        trials=trials,
        offsets=offsets,
        folds=folds,
        fold_payload=fold_payload,
        features=features,
        zero_scores=zero_scores,
        cache_manifest=cache_manifest,
        initial_state=initial_state,
        dataset_audit=dataset_audit,
        total_windows=total_windows,
    )


def split_subjects(folds: list[set[str]], fold_index: int) -> tuple[set[str], set[str], set[str]]:
    test_subjects = folds[fold_index]
    val_index = (fold_index + 1) % len(folds)
    val_subjects = folds[val_index]
    train_subjects = set().union(
        *[
            folds[index]
            for index in range(len(folds))
            if index not in (fold_index, val_index)
        ]
    )
    return train_subjects, val_subjects, test_subjects


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in core.DURATION_METRICS:
        values = np.asarray(
            [float(row[metric]) for row in rows if row.get(metric) is not None],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        output[metric] = {
            "mean": float(finite.mean()) if len(finite) else math.nan,
            "sample_std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
            "values": finite.tolist(),
        }
    return output


def legacy_integrity_check(
    calculated: dict[str, Any], stored: dict[str, Any], tolerance: float = 1e-6
) -> dict[str, Any]:
    keys = (
        "all_event_recall",
        "observable_event_recall",
        "mean_subject_false_alarms_per_hour",
        "pooled_false_alarms_per_hour",
        "worst_subject_false_alarms_per_hour",
        "cvar80_subject_false_alarms_per_hour",
        "proxy_median_delay_seconds",
        "proxy_p90_delay_seconds",
    )
    differences: dict[str, float] = {}
    for key in keys:
        left = float(calculated[key])
        right = float(stored[key])
        difference = abs(left - right)
        differences[key] = difference
        if difference > tolerance:
            raise RuntimeError(
                f"locked legacy metric mismatch for {key}: calculated={left} stored={right}"
            )
    return {"passed": True, "tolerance": tolerance, "absolute_differences": differences}


def load_legacy_head(path: Path) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["head_state_dict"] if "head_state_dict" in payload else payload
    head = core.new_head()
    head.load_state_dict(state, strict=True)
    return head


def run_locked_audit(args: argparse.Namespace, runtime: SimpleNamespace) -> dict[str, Any]:
    audit_dir = args.output_dir / "locked_alarm_duty_audit"
    records_dir = audit_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    fold_seed_records: list[dict[str, Any]] = []
    primary_zero_details: list[dict[str, Any]] = []
    primary_head_details: list[dict[str, Any]] = []
    primary_zero_summaries: list[dict[str, Any]] = []
    primary_head_summaries: list[dict[str, Any]] = []
    all_head_summaries: list[dict[str, Any]] = []

    for fold_index in range(5):
        _, val_subjects, test_subjects = split_subjects(runtime.folds, fold_index)
        val_trials, val_offsets, val_indices = runtime.legacy.subset(
            runtime.trials, runtime.offsets, val_subjects
        )
        test_trials, test_offsets, test_indices = runtime.legacy.subset(
            runtime.trials, runtime.offsets, test_subjects
        )
        del val_trials, val_offsets, val_indices
        zero_test_scores = np.asarray(runtime.zero_scores[test_indices], dtype=np.float32)
        for seed in core.SEEDS:
            core.atomic_json(
                progress_path,
                {
                    "state": "locked_alarm_duty_audit",
                    "dataset": args.dataset,
                    "fold": fold_index,
                    "seed": seed,
                },
            )
            legacy_dir = args.legacy_results / f"fold{fold_index}" / f"seed{seed}"
            stored_record = json.loads(
                (legacy_dir / "result.json").read_text(encoding="utf-8")
            )
            zero_threshold = float(stored_record["zero_shot"]["selected_threshold"])
            head_threshold = float(stored_record["head_tune"]["selected_threshold"])
            zero_summary, zero_details = core.evaluate_duration(
                runtime.external,
                test_trials,
                test_offsets,
                zero_test_scores,
                zero_threshold,
            )
            head = load_legacy_head(legacy_dir / "best_head.pth")
            head_scores = core.predict_head(head, runtime.features, test_indices)
            head_summary, head_details = core.evaluate_duration(
                runtime.external,
                test_trials,
                test_offsets,
                head_scores,
                head_threshold,
            )
            record = {
                "fold": fold_index,
                "seed": seed,
                "test_subjects": sorted(test_subjects),
                "frozen_weda_target_calibrated": {
                    "threshold": zero_threshold,
                    "summary": zero_summary,
                    "legacy_integrity": legacy_integrity_check(
                        zero_summary, stored_record["zero_shot"]["test"]
                    ),
                    "details": zero_details,
                },
                "standard_head_tune": {
                    "threshold": head_threshold,
                    "summary": head_summary,
                    "legacy_integrity": legacy_integrity_check(
                        head_summary, stored_record["head_tune"]["test"]
                    ),
                    "details": head_details,
                },
            }
            core.atomic_json(records_dir / f"fold{fold_index}_seed{seed}.json", record)
            fold_seed_records.append(
                {
                    "fold": fold_index,
                    "seed": seed,
                    "frozen_weda_target_calibrated": zero_summary,
                    "standard_head_tune": head_summary,
                }
            )
            all_head_summaries.append(head_summary)
            if seed == core.SEEDS[0]:
                primary_zero_details.extend(zero_details)
                primary_head_details.extend(head_details)
                primary_zero_summaries.append(zero_summary)
                primary_head_summaries.append(head_summary)
            print(
                f"AUDIT_FOLD_SEED dataset={args.dataset} fold={fold_index}/4 seed={seed} "
                f"zero_false_duty={zero_summary['pooled_false_alarm_duty_cycle']:.6f} "
                f"head_false_duty={head_summary['pooled_false_alarm_duty_cycle']:.6f}",
                flush=True,
            )
            del head

    core.atomic_json(
        progress_path,
        {
            "state": "locked_alarm_duty_bootstrap",
            "dataset": args.dataset,
            "iterations": args.bootstrap_iterations,
        },
    )
    zero_contributions = core.subject_contributions(primary_zero_details)
    head_contributions = core.subject_contributions(primary_head_details)
    subjects = sorted(zero_contributions)
    if subjects != sorted(head_contributions):
        raise RuntimeError("locked audit OOF subject mismatch")
    zero_oof = core.aggregate_subject_sample(zero_contributions, subjects)
    head_oof = core.aggregate_subject_sample(head_contributions, subjects)
    bootstrap = core.paired_bootstrap(
        primary_head_details,
        primary_zero_details,
        args.bootstrap_iterations,
        core.SEEDS[0],
        "standard_head_minus_frozen_weda",
    )
    final = {
        "version": "locked_alarm_duty_audit_v1_20260724",
        "dataset": args.dataset,
        "status": "complete",
        "analysis_type": "locked_no_retraining_no_rethresholding_duration_audit",
        "monitoring_denominator": "from_first_causal_score_to_trial_end",
        "event_external_duration_policy": (
            "alarm segments crossing the proxy event interval are split by overlap duration"
        ),
        "primary_seed": core.SEEDS[0],
        "frozen_weda_primary_oof": zero_oof,
        "standard_head_primary_oof": head_oof,
        "standard_head_all_fold_seed_runs": metric_summary(all_head_summaries),
        "paired_primary_oof_subject_bootstrap": bootstrap,
        "fold_seed_records": fold_seed_records,
        "cache_manifest": runtime.cache_manifest,
    }
    core.atomic_json(audit_dir / "locked_alarm_duty_audit.json", final)
    print(f"{args.dataset.upper()}_LOCKED_ALARM_DUTY_AUDIT_COMPLETE", flush=True)
    return final


def load_primary_standard_details(args: argparse.Namespace) -> list[dict[str, Any]]:
    records_dir = args.output_dir / "locked_alarm_duty_audit" / "records"
    details: list[dict[str, Any]] = []
    for fold_index in range(5):
        record = json.loads(
            (records_dir / f"fold{fold_index}_seed{core.SEEDS[0]}.json").read_text(
                encoding="utf-8"
            )
        )
        details.extend(record["standard_head_tune"]["details"])
    return details


def candidate_validation_key(record: dict[str, Any]) -> tuple[float, ...]:
    return core.threshold_sort_key(record["validation"], record["selected_threshold"])


def train_or_load_candidate(
    args: argparse.Namespace,
    runtime: SimpleNamespace,
    pool: core.TrainingPool,
    val_trials: list[Any],
    val_offsets: list[tuple[int, int]],
    val_indices: np.ndarray,
    profile: core.LossProfile,
    fold_index: int,
    seed: int,
    candidate_dir: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = candidate_dir / f"{profile.candidate_id}.pth"
    result_path = candidate_dir / f"{profile.candidate_id}.json"
    expected_config = training_config(args)
    if checkpoint_path.is_file() and result_path.is_file():
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if record.get("run_config") != expected_config:
            raise RuntimeError(
                f"resume configuration mismatch for {result_path}; use a new output directory"
            )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        head = core.new_head()
        head.load_state_dict(payload["head_state_dict"], strict=True)
        return head, record

    head, training = core.train_esra_candidate(
        runtime.features,
        pool,
        val_trials,
        val_offsets,
        val_indices,
        runtime.external,
        runtime.legacy.is_observable,
        runtime.initial_state,
        profile,
        seed,
        args.epochs,
        args.patience,
        args.steps_per_epoch,
        args.learning_rate,
        args.weight_decay,
        args.positive_bags_per_step,
        args.negative_bags_per_step,
        args.subjects_per_step,
        args.background_windows_per_subject,
        args.temporal_bags_per_step,
    )
    val_scores = core.predict_head(head, runtime.features, val_indices)
    try:
        threshold, validation, eligible = core.select_duration_threshold(
            runtime.external, val_trials, val_offsets, val_scores
        )
        status = "eligible"
    except RuntimeError as error:
        threshold = None
        validation = None
        eligible = 0
        status = f"ineligible: {error}"
    record = {
        "fold": fold_index,
        "seed": seed,
        "profile": asdict(profile),
        "candidate_id": profile.candidate_id,
        "run_config": expected_config,
        "status": status,
        "training": training,
        "selected_threshold": threshold,
        "eligible_thresholds": eligible,
        "validation": validation,
    }
    candidate_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "profile": asdict(profile),
            "fold": fold_index,
            "seed": seed,
        },
        checkpoint_path,
    )
    core.atomic_json(result_path, record)
    return head, record


def run_esra(args: argparse.Namespace, runtime: SimpleNamespace) -> dict[str, Any]:
    progress_path = args.output_dir / "progress.json"
    audit_path = (
        args.output_dir
        / "locked_alarm_duty_audit"
        / "locked_alarm_duty_audit.json"
    )
    if not audit_path.is_file():
        run_locked_audit(args, runtime)
    standard_primary_details = load_primary_standard_details(args)
    esra_dir = args.output_dir / "esra_ablation"
    run_records: list[dict[str, Any]] = []
    primary_details: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in args.variants
    }

    for fold_index in range(5):
        train_subjects, val_subjects, test_subjects = split_subjects(
            runtime.folds, fold_index
        )
        val_trials, val_offsets, val_indices = runtime.legacy.subset(
            runtime.trials, runtime.offsets, val_subjects
        )
        test_trials, test_offsets, test_indices = runtime.legacy.subset(
            runtime.trials, runtime.offsets, test_subjects
        )
        pool = core.build_training_pool(
            runtime.external,
            runtime.trials,
            runtime.offsets,
            train_subjects,
            runtime.legacy.is_observable,
        )
        for seed in core.SEEDS:
            for variant in args.variants:
                final_dir = esra_dir / f"fold{fold_index}" / f"seed{seed}" / variant
                final_path = final_dir / "result.json"
                if final_path.is_file():
                    record = json.loads(final_path.read_text(encoding="utf-8"))
                    if record.get("run_config") != training_config(args):
                        raise RuntimeError(
                            f"resume configuration mismatch for {final_path}; "
                            "use a new output directory"
                        )
                    run_records.append(record)
                    if seed == core.SEEDS[0] and record["status"] == "complete":
                        primary_details[variant].extend(record["test_details"])
                    print(
                        f"REUSING_ESRA_RUN dataset={args.dataset} fold={fold_index}/4 "
                        f"seed={seed} variant={variant}",
                        flush=True,
                    )
                    continue
                candidate_dir = final_dir / "candidates"
                candidate_records: list[dict[str, Any]] = []
                candidate_heads: dict[str, torch.nn.Module] = {}
                for candidate_index, profile in enumerate(core.candidate_grid(variant), start=1):
                    core.atomic_json(
                        progress_path,
                        {
                            "state": "training_esra_candidate",
                            "dataset": args.dataset,
                            "fold": fold_index,
                            "seed": seed,
                            "variant": variant,
                            "candidate": candidate_index,
                            "candidate_total": len(core.candidate_grid(variant)),
                            "candidate_id": profile.candidate_id,
                        },
                    )
                    head, candidate = train_or_load_candidate(
                        args,
                        runtime,
                        pool,
                        val_trials,
                        val_offsets,
                        val_indices,
                        profile,
                        fold_index,
                        seed,
                        candidate_dir,
                    )
                    candidate_records.append(candidate)
                    candidate_heads[profile.candidate_id] = head
                    print(
                        f"ESRA_CANDIDATE dataset={args.dataset} fold={fold_index}/4 seed={seed} "
                        f"variant={variant} candidate={profile.candidate_id} status={candidate['status']}",
                        flush=True,
                    )
                eligible = [
                    row for row in candidate_records if row["status"] == "eligible"
                ]
                if not eligible:
                    record = {
                        "fold": fold_index,
                        "seed": seed,
                        "variant": variant,
                        "status": "infeasible_no_validation_threshold",
                        "run_config": training_config(args),
                        "candidates": candidate_records,
                    }
                else:
                    selected = min(eligible, key=candidate_validation_key)
                    head = candidate_heads[selected["candidate_id"]]
                    test_scores = core.predict_head(head, runtime.features, test_indices)
                    test_summary, test_details = core.evaluate_duration(
                        runtime.external,
                        test_trials,
                        test_offsets,
                        test_scores,
                        float(selected["selected_threshold"]),
                    )
                    record = {
                        "fold": fold_index,
                        "seed": seed,
                        "variant": variant,
                        "status": "complete",
                        "run_config": training_config(args),
                        "train_subjects": sorted(train_subjects),
                        "validation_subjects": sorted(val_subjects),
                        "test_subjects": sorted(test_subjects),
                        "selected_candidate": selected,
                        "candidate_count": len(candidate_records),
                        "test": test_summary,
                        "test_details": test_details,
                    }
                    final_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "head_state_dict": head.state_dict(),
                            "selected_candidate": selected,
                            "fold": fold_index,
                            "seed": seed,
                            "variant": variant,
                        },
                        final_dir / "selected_head.pth",
                    )
                    if seed == core.SEEDS[0]:
                        primary_details[variant].extend(test_details)
                core.atomic_json(final_path, record)
                run_records.append(record)
                if record["status"] == "complete":
                    print(
                        f"COMPLETE_ESRA_FOLD_SEED dataset={args.dataset} fold={fold_index}/4 "
                        f"seed={seed} variant={variant} "
                        f"recall={record['test']['observable_event_recall']:.6f} "
                        f"false_duty={record['test']['pooled_false_alarm_duty_cycle']:.6f}",
                        flush=True,
                    )

    standard_contributions = core.subject_contributions(standard_primary_details)
    standard_subjects = sorted(standard_contributions)
    variants: dict[str, Any] = {}
    for variant in args.variants:
        completed = [
            row
            for row in run_records
            if row["variant"] == variant and row["status"] == "complete"
        ]
        primary = [row for row in completed if row["seed"] == core.SEEDS[0]]
        payload: dict[str, Any] = {
            "completed_runs": len(completed),
            "expected_runs": 15,
            "all_fold_seed_runs": metric_summary([row["test"] for row in completed])
            if completed
            else None,
        }
        if len(primary) == 5:
            contributions = core.subject_contributions(primary_details[variant])
            subjects = sorted(contributions)
            if subjects != standard_subjects:
                raise RuntimeError(f"OOF subject mismatch for {variant}")
            payload["primary_oof"] = core.aggregate_subject_sample(contributions, subjects)
            payload["paired_vs_standard_head_bootstrap"] = core.paired_bootstrap(
                primary_details[variant],
                standard_primary_details,
                args.bootstrap_iterations,
                core.SEEDS[0],
                f"{variant}_minus_standard_head",
            )
        else:
            payload["primary_oof"] = None
            payload["paired_vs_standard_head_bootstrap"] = None
        variants[variant] = payload

    final = {
        "version": "esra_head_v6_20260724",
        "dataset": args.dataset,
        "status": "complete",
        "analysis_type": "development_benchmark_not_untouched_confirmation",
        "encoder": "frozen_unimts",
        "trainable_head_parameters": 2050,
        "target_validation_observable_recall": core.TARGET_RECALL,
        "threshold_selection_primary": "mean_subject_false_alarm_duty_cycle",
        "primary_seed": core.SEEDS[0],
        "sensitivity_seeds": list(core.SEEDS[1:]),
        "variants": variants,
        "run_records": [
            {
                key: value
                for key, value in row.items()
                if key not in ("test_details",)
            }
            for row in run_records
        ],
        "cache_manifest": runtime.cache_manifest,
    }
    core.atomic_json(esra_dir / "esra_results.json", final)
    print(f"{args.dataset.upper()}_ESRA_HEAD_COMPLETE", flush=True)
    return final


def main() -> None:
    args = resolve_args(parse_args())
    core.atomic_json(
        args.output_dir / "progress.json",
        {"state": "loading_runtime", "dataset": args.dataset, "mode": args.mode},
    )
    runtime = load_runtime(args)
    core.atomic_json(
        args.output_dir / "runtime_audit.json",
        {
            "dataset": args.dataset,
            "dataset_audit": runtime.dataset_audit,
            "fold_lock": runtime.fold_payload,
            "total_windows": runtime.total_windows,
            "cache_manifest": runtime.cache_manifest,
            "protocol_sha256": runtime.legacy.sha256(args.protocol),
        },
    )
    if args.mode in ("audit", "all"):
        run_locked_audit(args, runtime)
    if args.mode in ("esra", "all"):
        run_esra(args, runtime)
    core.atomic_json(
        args.output_dir / "progress.json",
        {"state": "complete", "dataset": args.dataset, "mode": args.mode},
    )
    print(f"{args.dataset.upper()}_FALL_ESRA_V6_COMPLETE mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()

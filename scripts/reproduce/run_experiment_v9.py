from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import torch

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
LINEAR_EVENT_SRC = REPO_ROOT / "src" / "linear_event"
if str(LINEAR_EVENT_SRC) not in sys.path:
    sys.path.insert(0, str(LINEAR_EVENT_SRC))

import linear_event_core_v9 as linear

SEEDS = (20260724, 20260725, 20260726)
VARIANTS = ("linear_event_mil", "linear_event_duty")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-selected linear event adapter experiment."
    )
    parser.add_argument("--dataset", choices=("smartfallmm", "umafall"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--v6-code-dir", type=Path, required=True)
    parser.add_argument("--v6-results-dir", type=Path, required=True)
    parser.add_argument("--external-code-dir", type=Path, required=True)
    parser.add_argument("--gate2a-code-dir", type=Path, required=True)
    parser.add_argument("--unimts-code-dir", type=Path, required=True)
    parser.add_argument("--released-checkpoint", type=Path, required=True)
    parser.add_argument("--weda-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps-per-epoch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-bags-per-step", type=int, default=8)
    parser.add_argument("--negative-bags-per-step", type=int, default=8)
    parser.add_argument("--subjects-per-step", type=int, default=8)
    parser.add_argument("--background-windows-per-subject", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    return parser.parse_args()


def resolved(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "dataset_root",
        "v6_code_dir",
        "v6_results_dir",
        "external_code_dir",
        "gate2a_code_dir",
        "unimts_code_dir",
        "released_checkpoint",
        "weda_checkpoint",
        "cache_dir",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    for path in (
        args.dataset_root,
        args.v6_code_dir,
        args.v6_results_dir,
        args.external_code_dir,
        args.gate2a_code_dir,
        args.unimts_code_dir,
        args.cache_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (args.released_checkpoint, args.weda_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.epochs < 1 or args.steps_per_epoch < 1:
        raise ValueError("epochs and steps-per-epoch must be positive")
    if args.bootstrap_iterations < 1000:
        raise ValueError("bootstrap iterations must be at least 1000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def load_v6(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(args.v6_code_dir))
    v6_runner = importlib.import_module("run_experiment")
    v6_core = importlib.import_module("esra_core")
    v6_args = SimpleNamespace(
        dataset=args.dataset,
        mode="esra",
        dataset_root=args.dataset_root,
        external_code_dir=args.external_code_dir,
        gate2a_code_dir=args.gate2a_code_dir,
        unimts_code_dir=args.unimts_code_dir,
        released_checkpoint=args.released_checkpoint,
        weda_checkpoint=args.weda_checkpoint,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        device=args.device,
        feature_batch_size=args.feature_batch_size,
        bootstrap_iterations=args.bootstrap_iterations,
        epochs=args.epochs,
        patience=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        positive_bags_per_step=args.positive_bags_per_step,
        negative_bags_per_step=args.negative_bags_per_step,
        subjects_per_step=args.subjects_per_step,
        background_windows_per_subject=args.background_windows_per_subject,
        temporal_bags_per_step=1,
        variants=["event_mil"],
    )
    v6_args = v6_runner.resolve_args(v6_args)
    runtime = v6_runner.load_runtime(v6_args)
    return v6_runner, v6_core, runtime


def config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": "fall_linear_event_adapter_v9_20260728",
        "protocol_sha256": sha256(REPO_ROOT / "docs" / "protocols" / "PROTOCOL_V9.md"),
        "dataset": args.dataset,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "positive_bags_per_step": args.positive_bags_per_step,
        "negative_bags_per_step": args.negative_bags_per_step,
        "subjects_per_step": args.subjects_per_step,
        "background_windows_per_subject": args.background_windows_per_subject,
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
    }


def write_progress(v6_core: Any, args: argparse.Namespace, payload: dict[str, Any]) -> None:
    v6_core.atomic_json(args.output_dir / "progress.json", payload)


def load_reference_details(
    args: argparse.Namespace, method: str
) -> list[dict[str, Any]]:
    root = args.v6_results_dir / args.dataset
    details: list[dict[str, Any]] = []
    for fold in range(5):
        if method == "standard_head":
            path = (
                root
                / "locked_alarm_duty_audit"
                / "records"
                / f"fold{fold}_seed{SEEDS[0]}.json"
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            details.extend(record["standard_head_tune"]["details"])
        elif method == "esra_full":
            path = (
                root
                / "esra_ablation"
                / f"fold{fold}"
                / f"seed{SEEDS[0]}"
                / "esra_full"
                / "result.json"
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            details.extend(record["test_details"])
        else:
            raise ValueError(method)
    return details


def fit_or_load_probe(
    args: argparse.Namespace,
    v6_core: Any,
    runtime: Any,
    fold: int,
    train_subjects: set[str],
    val_subjects: set[str],
    test_subjects: set[str],
) -> tuple[Any, dict[str, Any]]:
    fold_dir = args.output_dir / f"fold{fold}" / "linear_probe"
    result_path = fold_dir / "result.json"
    model_path = fold_dir / "model.joblib"
    if result_path.is_file() and model_path.is_file():
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if record["run_config"] != config(args):
            raise RuntimeError(f"resume configuration mismatch: {result_path}")
        return joblib.load(model_path), record

    val_trials, val_offsets, val_indices = runtime.legacy.subset(
        runtime.trials, runtime.offsets, val_subjects
    )
    test_trials, test_offsets, test_indices = runtime.legacy.subset(
        runtime.trials, runtime.offsets, test_subjects
    )
    train_indices, train_labels, training_windows = linear.all_training_indices(
        runtime.external,
        runtime.trials,
        runtime.offsets,
        train_subjects,
        runtime.legacy.is_observable,
    )
    model, selected, candidates = linear.fit_probe_candidates(
        v6_core,
        runtime.external,
        runtime.features,
        train_indices,
        train_labels,
        val_trials,
        val_offsets,
        val_indices,
    )
    head = linear.head_from_pipeline(model)
    direct_scores = model.predict_proba(
        np.asarray(runtime.features[test_indices], dtype=np.float32)
    )[:, 1].astype(np.float32)
    head_scores = linear.predict(head, runtime.features, test_indices)
    agreement = float(np.max(np.abs(direct_scores - head_scores)))
    if agreement > 2e-5:
        raise RuntimeError(f"sklearn/torch probe mismatch: {agreement}")
    test, test_details = v6_core.evaluate_duration(
        runtime.external,
        test_trials,
        test_offsets,
        head_scores,
        float(selected["selected_threshold"]),
    )
    record = {
        "fold": fold,
        "method": "linear_probe",
        "status": "complete",
        "run_config": config(args),
        "train_subjects": sorted(train_subjects),
        "validation_subjects": sorted(val_subjects),
        "test_subjects": sorted(test_subjects),
        "training_windows": training_windows,
        "selected_candidate": selected,
        "candidates": candidates,
        "trainable_parameters": 513,
        "sklearn_torch_max_abs_probability_difference": agreement,
        "test": test,
        "test_details": test_details,
    }
    fold_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    v6_core.atomic_json(result_path, record)
    return model, record


def train_or_load_candidate(
    args: argparse.Namespace,
    v6_core: Any,
    runtime: Any,
    pool: Any,
    val_trials: list[Any],
    val_offsets: list[tuple[int, int]],
    val_indices: np.ndarray,
    probe: Any,
    profile: linear.Profile,
    fold: int,
    seed: int,
    candidate_dir: Path,
) -> tuple[linear.LinearEventHead, dict[str, Any]]:
    result_path = candidate_dir / f"{profile.candidate_id}.json"
    checkpoint_path = candidate_dir / f"{profile.candidate_id}.pth"
    if result_path.is_file() and checkpoint_path.is_file():
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if record["run_config"] != config(args):
            raise RuntimeError(f"resume configuration mismatch: {result_path}")
        head = linear.head_from_pipeline(probe)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        head.load_state_dict(payload["head_state_dict"], strict=True)
        return head, record

    head, training = linear.train_candidate(
        v6_core,
        runtime.features,
        pool,
        val_trials,
        val_offsets,
        val_indices,
        runtime.external,
        runtime.legacy.is_observable,
        probe,
        profile,
        seed,
        args.epochs,
        args.steps_per_epoch,
        args.learning_rate,
        args.weight_decay,
        args.positive_bags_per_step,
        args.negative_bags_per_step,
        args.subjects_per_step,
        args.background_windows_per_subject,
    )
    record = {
        "fold": fold,
        "seed": seed,
        "profile": asdict(profile),
        "candidate_id": profile.candidate_id,
        "status": "eligible",
        "run_config": config(args),
        "training": training,
        "selected_threshold": training["selected_threshold"],
        "validation": training["validation"],
    }
    candidate_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "profile": asdict(profile),
            "fold": fold,
            "seed": seed,
        },
        checkpoint_path,
    )
    v6_core.atomic_json(result_path, record)
    return head, record


def candidate_key(v6_core: Any, record: dict[str, Any]) -> tuple[float, ...]:
    return v6_core.threshold_sort_key(
        record["validation"], float(record["selected_threshold"])
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    v6_runner, v6_core, runtime = load_v6(args)
    records: list[dict[str, Any]] = []
    for fold in range(5):
        train_subjects, val_subjects, test_subjects = v6_runner.split_subjects(
            runtime.folds, fold
        )
        probe, probe_record = fit_or_load_probe(
            args,
            v6_core,
            runtime,
            fold,
            train_subjects,
            val_subjects,
            test_subjects,
        )
        print(
            f"COMPLETE_LINEAR_PROBE dataset={args.dataset} fold={fold}/4 "
            f"recall={probe_record['test']['observable_event_recall']:.6f} "
            f"duty={probe_record['test']['mean_subject_false_alarm_duty_cycle']:.6f}",
            flush=True,
        )
        val_trials, val_offsets, val_indices = runtime.legacy.subset(
            runtime.trials, runtime.offsets, val_subjects
        )
        test_trials, test_offsets, test_indices = runtime.legacy.subset(
            runtime.trials, runtime.offsets, test_subjects
        )
        pool = v6_core.build_training_pool(
            runtime.external,
            runtime.trials,
            runtime.offsets,
            train_subjects,
            runtime.legacy.is_observable,
        )
        for seed in SEEDS:
            for variant in VARIANTS:
                final_dir = (
                    args.output_dir
                    / f"fold{fold}"
                    / variant
                    / f"seed{seed}"
                )
                final_path = final_dir / "result.json"
                if final_path.is_file():
                    record = json.loads(final_path.read_text(encoding="utf-8"))
                    if record["run_config"] != config(args):
                        raise RuntimeError(
                            f"resume configuration mismatch: {final_path}"
                        )
                    records.append(record)
                    print(
                        f"REUSING_LINEAR_EVENT dataset={args.dataset} fold={fold}/4 "
                        f"seed={seed} variant={variant}",
                        flush=True,
                    )
                    continue
                candidate_dir = final_dir / "candidates"
                candidates: list[dict[str, Any]] = []
                heads: dict[str, linear.LinearEventHead] = {}
                profiles = linear.candidate_grid(variant)
                for index, profile in enumerate(profiles, start=1):
                    write_progress(
                        v6_core,
                        args,
                        {
                            "state": "training_linear_event_candidate",
                            "dataset": args.dataset,
                            "fold": fold,
                            "seed": seed,
                            "variant": variant,
                            "candidate": index,
                            "candidate_total": len(profiles),
                            "candidate_id": profile.candidate_id,
                        },
                    )
                    head, candidate = train_or_load_candidate(
                        args,
                        v6_core,
                        runtime,
                        pool,
                        val_trials,
                        val_offsets,
                        val_indices,
                        probe,
                        profile,
                        fold,
                        seed,
                        candidate_dir,
                    )
                    candidates.append(candidate)
                    heads[profile.candidate_id] = head
                selected = min(
                    candidates, key=lambda row: candidate_key(v6_core, row)
                )
                head = heads[selected["candidate_id"]]
                test_scores = linear.predict(
                    head, runtime.features, test_indices
                )
                test, test_details = v6_core.evaluate_duration(
                    runtime.external,
                    test_trials,
                    test_offsets,
                    test_scores,
                    float(selected["selected_threshold"]),
                )
                record = {
                    "fold": fold,
                    "seed": seed,
                    "variant": variant,
                    "status": "complete",
                    "run_config": config(args),
                    "train_subjects": sorted(train_subjects),
                    "validation_subjects": sorted(val_subjects),
                    "test_subjects": sorted(test_subjects),
                    "selected_candidate": selected,
                    "candidates": candidates,
                    "trainable_parameters": 513,
                    "test": test,
                    "test_details": test_details,
                }
                final_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "head_state_dict": head.state_dict(),
                        "selected_candidate": selected,
                    },
                    final_dir / "selected_head.pth",
                )
                v6_core.atomic_json(final_path, record)
                records.append(record)
                print(
                    f"COMPLETE_LINEAR_EVENT dataset={args.dataset} fold={fold}/4 "
                    f"seed={seed} variant={variant} "
                    f"recall={test['observable_event_recall']:.6f} "
                    f"duty={test['mean_subject_false_alarm_duty_cycle']:.6f}",
                    flush=True,
                )

    primary_details: dict[str, list[dict[str, Any]]] = {
        "linear_probe": [],
        **{variant: [] for variant in VARIANTS},
    }
    seed_details: dict[str, dict[int, list[dict[str, Any]]]] = {
        variant: {seed: [] for seed in SEEDS} for variant in VARIANTS
    }
    for fold in range(5):
        probe_record = json.loads(
            (
                args.output_dir
                / f"fold{fold}"
                / "linear_probe"
                / "result.json"
            ).read_text(encoding="utf-8")
        )
        primary_details["linear_probe"].extend(probe_record["test_details"])
        for variant in VARIANTS:
            for seed in SEEDS:
                record = json.loads(
                    (
                        args.output_dir
                        / f"fold{fold}"
                        / variant
                        / f"seed{seed}"
                        / "result.json"
                    ).read_text(encoding="utf-8")
                )
                seed_details[variant][seed].extend(record["test_details"])
                if seed == SEEDS[0]:
                    primary_details[variant].extend(record["test_details"])

    primary_oof = {
        method: v6_core.aggregate_subject_sample(
            v6_core.subject_contributions(details),
            sorted(v6_core.subject_contributions(details)),
        )
        for method, details in primary_details.items()
    }
    sensitivity = {
        variant: {
            str(seed): v6_core.aggregate_subject_sample(
                v6_core.subject_contributions(seed_details[variant][seed]),
                sorted(v6_core.subject_contributions(seed_details[variant][seed])),
            )
            for seed in SEEDS
        }
        for variant in VARIANTS
    }
    standard_details = load_reference_details(args, "standard_head")
    esra_details = load_reference_details(args, "esra_full")
    comparisons = {
        "linear_event_duty_minus_linear_probe": v6_core.paired_bootstrap(
            primary_details["linear_event_duty"],
            primary_details["linear_probe"],
            args.bootstrap_iterations,
            20260728,
            "linear_event_duty_minus_linear_probe",
        ),
        "linear_event_duty_minus_standard_head": v6_core.paired_bootstrap(
            primary_details["linear_event_duty"],
            standard_details,
            args.bootstrap_iterations,
            20260728,
            "linear_event_duty_minus_standard_head",
        ),
        "linear_event_duty_minus_esra_full": v6_core.paired_bootstrap(
            primary_details["linear_event_duty"],
            esra_details,
            args.bootstrap_iterations,
            20260728,
            "linear_event_duty_minus_esra_full",
        ),
    }
    final = {
        "version": "fall_linear_event_adapter_v9_20260728",
        "state": "complete",
        "analysis_type": "subject_independent_linear_event_adapter",
        "dataset": args.dataset,
        "confirmatory_status": "developmental_not_untouched",
        "primary_seed": SEEDS[0],
        "sensitivity_seeds": list(SEEDS[1:]),
        "trainable_parameters": 513,
        "primary_oof": primary_oof,
        "seed_sensitivity": sensitivity,
        "paired_bootstrap": comparisons,
        "run_config": config(args),
    }
    v6_core.atomic_json(args.output_dir / "linear_event_results.json", final)
    write_progress(v6_core, args, {"state": "complete"})
    print(f"{args.dataset.upper()}_LINEAR_EVENT_V9_COMPLETE", flush=True)
    return final


def main() -> None:
    args = resolved(parse_args())
    run(args)


if __name__ == "__main__":
    main()

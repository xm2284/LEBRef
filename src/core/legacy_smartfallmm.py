from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset


SEEDS = (20260724, 20260725, 20260726)
TARGET_RECALL = 0.90
METRICS = (
    "all_event_recall",
    "observable_event_recall",
    "observable_coverage",
    "mean_subject_false_alarms_per_hour",
    "pooled_false_alarms_per_hour",
    "worst_subject_false_alarms_per_hour",
    "cvar80_subject_false_alarms_per_hour",
    "proxy_median_delay_seconds",
    "proxy_p90_delay_seconds",
)
BOOTSTRAP_METRICS = tuple(metric for metric in METRICS if metric != "observable_coverage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Version-locked SmartFallMM five-fold Head Tune replication."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--external-code-dir", type=Path, required=True)
    parser.add_argument("--gate2a-code-dir", type=Path, required=True)
    parser.add_argument("--unimts-code-dir", type=Path, required=True)
    parser.add_argument("--released-checkpoint", type=Path, required=True)
    parser.add_argument("--weda-checkpoint", type=Path, required=True)
    parser.add_argument("--fold-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(clean_json(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_imports(args: argparse.Namespace) -> tuple[Any, Any]:
    for path in (args.external_code_dir, args.gate2a_code_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_locked_external as external
    import weda_common

    return external, weda_common


def load_fold_lock(path: Path, observed: set[str]) -> tuple[list[set[str]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = [set(map(str, row["young"] + row["old"])) for row in payload["folds"]]
    if len(folds) != 5:
        raise RuntimeError("five folds are required")
    if any(folds[i] & folds[j] for i in range(5) for j in range(i + 1, 5)):
        raise RuntimeError("subject overlap in fold lock")
    locked = set().union(*folds)
    if locked != observed:
        raise RuntimeError(
            f"fold/data mismatch: missing={sorted(observed - locked)} extra={sorted(locked - observed)}"
        )
    return folds, payload


def trial_ends(external: Any, trial: Any) -> np.ndarray:
    length = len(trial.signal)
    if length < external.WINDOW_LENGTH:
        return np.asarray([length - 1], dtype=np.int64)
    ends = np.arange(
        external.WINDOW_LENGTH - 1,
        length,
        external.STRIDE_SAMPLES,
        dtype=np.int64,
    )
    if ends[-1] != length - 1:
        ends = np.concatenate([ends, np.asarray([length - 1], dtype=np.int64)])
    return ends


def attach_offsets(external: Any, trials: list[Any]) -> tuple[list[tuple[int, int]], int]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for trial in trials:
        ends = trial_ends(external, trial)
        trial.score_times = ends.astype(np.float64) / external.SAMPLE_RATE_HZ
        offsets.append((cursor, cursor + len(ends)))
        cursor += len(ends)
    return offsets, cursor


def is_observable(external: Any, trial: Any) -> bool:
    if not trial.is_fall:
        return False
    detection_end = min(
        float(trial.duration_seconds),
        float(trial.proxy_end) + float(external.GRACE_SECONDS),
    )
    return float(trial.score_times[0]) <= detection_end


def build_model(args: argparse.Namespace, weda_common: Any, device: torch.device) -> nn.Module:
    return weda_common.build_model_from_checkpoint(
        variant="acc",
        checkpoint_path=args.weda_checkpoint,
        unimts_code_dir=args.unimts_code_dir,
        released_checkpoint=args.released_checkpoint,
        device=device,
    )


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "features": cache_dir / "encoder_features.float32.npy",
        "zero_scores": cache_dir / "zero_shot_scores.float32.npy",
        "head": cache_dir / "initial_head_state.pth",
        "manifest": cache_dir / "cache_manifest.json",
    }


def iter_causal_window_batches(
    external: Any,
    trials: list[Any],
    offsets: list[tuple[int, int]],
    batch_size: int,
):
    if batch_size < 1:
        raise ValueError("feature batch size must be positive")
    pending: deque[np.ndarray] = deque()
    pending_count = 0

    def take(count: int) -> np.ndarray:
        nonlocal pending_count
        pieces: list[np.ndarray] = []
        remaining = count
        while remaining:
            current = pending[0]
            current_count = len(current)
            used = min(remaining, current_count)
            pieces.append(current[:used])
            if used == current_count:
                pending.popleft()
            else:
                pending[0] = current[used:]
            pending_count -= used
            remaining -= used
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)

    for trial_index, (trial, (start, end)) in enumerate(zip(trials, offsets), start=1):
        windows, ends = external.causal_windows(trial.signal)
        expected_times = ends.astype(np.float64) / external.SAMPLE_RATE_HZ
        if not np.array_equal(expected_times, trial.score_times):
            raise RuntimeError("feature-cache score-time mismatch")
        if len(windows) != end - start:
            raise RuntimeError("feature-cache trial offset mismatch")
        pending.append(windows)
        pending_count += len(windows)
        while pending_count >= batch_size:
            yield take(batch_size), trial_index
    if pending_count:
        yield take(pending_count), len(trials)


def create_or_load_cache(
    args: argparse.Namespace,
    external: Any,
    weda_common: Any,
    trials: list[Any],
    offsets: list[tuple[int, int]],
    total_windows: int,
    dataset_digest: str,
    progress_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    paths = cache_paths(args.cache_dir)
    if all(path.is_file() for path in paths.values()):
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("version") != "smartfallmm_unimts_feature_cache_v1_20260724":
            raise RuntimeError("feature cache version mismatch")
        if manifest["total_windows"] != total_windows:
            raise RuntimeError("feature cache window count mismatch")
        if manifest["dataset_model_input_sha256"] != dataset_digest:
            raise RuntimeError("feature cache dataset digest mismatch")
        if manifest.get("released_checkpoint_sha256") != sha256(args.released_checkpoint):
            raise RuntimeError("feature cache released checkpoint mismatch")
        if manifest.get("weda_checkpoint_sha256") != sha256(args.weda_checkpoint):
            raise RuntimeError("feature cache WEDA checkpoint mismatch")
        features = np.load(paths["features"], mmap_mode="r")
        zero_scores = np.load(paths["zero_scores"], mmap_mode="r")
        if (
            features.shape != (total_windows, 512)
            or zero_scores.shape != (total_windows,)
            or features.dtype != np.float32
            or zero_scores.dtype != np.float32
        ):
            raise RuntimeError("feature cache shape mismatch")
        print("REUSING_FEATURE_CACHE", flush=True)
        return features, zero_scores, manifest

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_features = args.cache_dir / "encoder_features.partial.npy"
    temporary_scores = args.cache_dir / "zero_shot_scores.partial.npy"
    features = np.lib.format.open_memmap(
        temporary_features, mode="w+", dtype=np.float32, shape=(total_windows, 512)
    )
    zero_scores = np.lib.format.open_memmap(
        temporary_scores, mode="w+", dtype=np.float32, shape=(total_windows,)
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    model = build_model(args, weda_common, device)
    model.eval()
    torch.save({key: value.detach().cpu() for key, value in model.head.state_dict().items()}, paths["head"])
    started = time.time()
    atomic_json(
        progress_path,
        {
            "state": "caching_features",
            "prepared_trials": 0,
            "total_trials": len(trials),
            "completed_windows": 0,
            "total_windows": total_windows,
        },
    )
    cursor = 0
    last_report_trial = 0
    with torch.inference_mode():
        for windows, prepared_trials in iter_causal_window_batches(
            external, trials, offsets, args.feature_batch_size
        ):
            batch = torch.from_numpy(windows).to(device, non_blocking=True)
            encoded = model.features(batch, 0)
            logits = model.head(encoded)
            count = len(batch)
            features[cursor : cursor + count] = encoded.float().cpu().numpy()
            zero_scores[cursor : cursor + count] = (
                torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
            )
            cursor += count
            if (
                prepared_trials - last_report_trial >= 100
                or prepared_trials == len(trials)
            ):
                atomic_json(
                    progress_path,
                    {
                        "state": "caching_features",
                        "prepared_trials": prepared_trials,
                        "total_trials": len(trials),
                        "completed_windows": cursor,
                        "total_windows": total_windows,
                        "elapsed_seconds": time.time() - started,
                    },
                )
                print(
                    f"CACHE_FEATURES trials={prepared_trials}/{len(trials)} "
                    f"windows={cursor}/{total_windows}",
                    flush=True,
                )
                last_report_trial = prepared_trials
    if cursor != total_windows:
        raise RuntimeError("feature-cache total window mismatch")
    features.flush()
    zero_scores.flush()
    del features, zero_scores, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    os.replace(temporary_features, paths["features"])
    os.replace(temporary_scores, paths["zero_scores"])
    manifest = {
        "version": "smartfallmm_unimts_feature_cache_v1_20260724",
        "total_windows": total_windows,
        "feature_shape": [total_windows, 512],
        "feature_dtype": "float32",
        "dataset_model_input_sha256": dataset_digest,
        "released_checkpoint_sha256": sha256(args.released_checkpoint),
        "weda_checkpoint_sha256": sha256(args.weda_checkpoint),
        "initial_head_parameters": 2050,
    }
    atomic_json(paths["manifest"], manifest)
    return (
        np.load(paths["features"], mmap_mode="r"),
        np.load(paths["zero_scores"], mmap_mode="r"),
        manifest,
    )


def subset(
    trials: list[Any],
    offsets: list[tuple[int, int]],
    subjects: set[str],
) -> tuple[list[Any], list[tuple[int, int]], np.ndarray]:
    selected_trials: list[Any] = []
    local_offsets: list[tuple[int, int]] = []
    chunks: list[np.ndarray] = []
    cursor = 0
    for trial, (start, end) in zip(trials, offsets):
        if trial.subject not in subjects:
            continue
        indices = np.arange(start, end, dtype=np.int64)
        selected_trials.append(trial)
        local_offsets.append((cursor, cursor + len(indices)))
        chunks.append(indices)
        cursor += len(indices)
    return selected_trials, local_offsets, np.concatenate(chunks)


def labels_for_trials(trials: list[Any]) -> np.ndarray:
    labels: list[np.ndarray] = []
    for trial in trials:
        value = np.zeros(len(trial.score_times), dtype=np.int64)
        if trial.is_fall:
            value[
                (trial.score_times >= float(trial.proxy_start))
                & (trial.score_times <= float(trial.proxy_end))
            ] = 1
        labels.append(value)
    return np.concatenate(labels)


def training_indices(
    external: Any,
    trials: list[Any],
    offsets: list[tuple[int, int]],
    subjects: set[str],
    negative_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    positive: list[int] = []
    negative: list[int] = []
    excluded_unobservable = 0
    for trial, (start, end) in zip(trials, offsets):
        if trial.subject not in subjects:
            continue
        indices = np.arange(start, end, dtype=np.int64)
        if trial.is_fall:
            if not is_observable(external, trial):
                excluded_unobservable += 1
                continue
            mask = (
                (trial.score_times >= float(trial.proxy_start))
                & (trial.score_times <= float(trial.proxy_end))
            )
            positive.extend(indices[mask].tolist())
        else:
            negative.extend(indices.tolist())
    if not positive or not negative:
        raise RuntimeError("training requires positive and negative windows")
    rng = random.Random(seed)
    target_negative = min(len(negative), int(math.ceil(len(positive) * negative_ratio)))
    selected_negative = rng.sample(negative, target_negative)
    combined = [(index, 1) for index in positive] + [
        (index, 0) for index in selected_negative
    ]
    rng.shuffle(combined)
    return (
        np.asarray([row[0] for row in combined], dtype=np.int64),
        np.asarray([row[1] for row in combined], dtype=np.int64),
        {
            "positive": len(positive),
            "negative_available": len(negative),
            "negative_sampled": target_negative,
            "total_sampled": len(combined),
            "excluded_unobservable_fall_trials": excluded_unobservable,
        },
    )


def new_head() -> nn.Module:
    return nn.Sequential(nn.LayerNorm(512), nn.Dropout(0.1), nn.Linear(512, 2))


def predict_head(
    head: nn.Module,
    features: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            values = np.asarray(features[indices[start : start + batch_size]], dtype=np.float32)
            batch = torch.from_numpy(values).to(device)
            outputs.append(torch.softmax(head(batch), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def train_head(
    args: argparse.Namespace,
    features: np.ndarray,
    train_indices: np.ndarray,
    train_labels: np.ndarray,
    val_indices: np.ndarray,
    val_labels: np.ndarray,
    initial_state: dict[str, Any],
    seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    head = new_head().to(device)
    head.load_state_dict(initial_state, strict=True)
    train_values = np.asarray(features[train_indices], dtype=np.float32)
    dataset = TensorDataset(torch.from_numpy(train_values), torch.from_numpy(train_labels))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    counts = np.bincount(train_labels, minlength=2)
    weights = len(train_labels) / (2.0 * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32), label_smoothing=0.02
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    best_ap = -math.inf
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
        scheduler.step()
        scores = predict_head(head, features, val_indices, device)
        val_ap = float(average_precision_score(val_labels, scores))
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_labels),
                "validation_window_pr_auc": val_ap,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if val_ap > best_ap + 1e-8:
            best_ap = val_ap
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("head training produced no checkpoint")
    head.load_state_dict(best_state, strict=True)
    return head, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_window_pr_auc": best_ap,
        "epochs_run": len(history),
        "trainable_parameters": sum(p.numel() for p in head.parameters()),
        "history": history,
    }


def evaluate(
    external: Any,
    trials: list[Any],
    offsets: list[tuple[int, int]],
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    params = external.Params(threshold, threshold, 1, 2.0)
    details: list[dict[str, Any]] = []
    for trial, (start, end) in zip(trials, offsets):
        row = external.evaluate_trial(trial, scores[start:end], params)
        row["observable"] = int(is_observable(external, trial))
        details.append(row)
    summary = external.summarize_trials(details, params)
    falls = [row for row in details if row["is_fall"] == 1]
    observable = [row for row in falls if row["observable"] == 1]
    summary.update(
        {
            "all_event_recall": summary["impact_proxy_event_recall"],
            "observable_event_recall": sum(int(row["detected"]) for row in observable)
            / len(observable),
            "observable_coverage": len(observable) / len(falls),
            "all_fall_events": len(falls),
            "observable_fall_events": len(observable),
            "detected_observable_fall_events": sum(
                int(row["detected"]) for row in observable
            ),
        }
    )
    return summary, details


def select_threshold(
    external: Any,
    trials: list[Any],
    offsets: list[tuple[int, int]],
    scores: np.ndarray,
) -> tuple[float, dict[str, Any], int]:
    evaluated = [
        (float(threshold), evaluate(external, trials, offsets, scores, float(threshold))[0])
        for threshold in np.linspace(0.01, 0.99, 99)
    ]
    eligible = [row for row in evaluated if row[1]["observable_event_recall"] >= TARGET_RECALL]
    if not eligible:
        best = max(row[1]["observable_event_recall"] for row in evaluated)
        raise RuntimeError(f"no threshold reaches target observable recall; best={best}")
    selected = min(
        eligible,
        key=lambda row: (
            row[1]["mean_subject_false_alarms_per_hour"],
            row[1]["worst_subject_false_alarms_per_hour"],
            row[1]["cvar80_subject_false_alarms_per_hour"],
            row[1]["proxy_median_delay_seconds"],
            -row[0],
        ),
    )
    return selected[0], selected[1], len(eligible)


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in records], dtype=np.float64)
        output[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    return output


def subject_contributions(details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        groups[str(row["subject"])].append(row)
    output: dict[str, dict[str, Any]] = {}
    for subject, rows in groups.items():
        falls = [row for row in rows if row["is_fall"] == 1]
        observable = [row for row in falls if row["observable"] == 1]
        hours = sum(float(row["non_event_hours"]) for row in rows)
        output[subject] = {
            "false_alarms": sum(float(row["false_alarms"]) for row in rows),
            "hours": hours,
            "all_falls": len(falls),
            "detected_all": sum(int(row["detected"]) for row in falls),
            "observable_falls": len(observable),
            "detected_observable": sum(int(row["detected"]) for row in observable),
            "delays": [
                float(row["proxy_delay_seconds"])
                for row in falls
                if row["proxy_delay_seconds"] is not None
            ],
        }
    return output


def aggregate_subject_sample(
    external: Any, contributions: dict[str, dict[str, Any]], sampled: list[str]
) -> dict[str, float]:
    rows = [contributions[subject] for subject in sampled]
    rates = np.asarray([row["false_alarms"] / row["hours"] for row in rows])
    all_falls = sum(row["all_falls"] for row in rows)
    observable = sum(row["observable_falls"] for row in rows)
    delays = [value for row in rows for value in row["delays"]]
    return {
        "all_event_recall": sum(row["detected_all"] for row in rows) / all_falls
        if all_falls
        else math.nan,
        "observable_event_recall": sum(row["detected_observable"] for row in rows)
        / observable
        if observable
        else math.nan,
        "observable_coverage": observable / all_falls if all_falls else math.nan,
        "mean_subject_false_alarms_per_hour": float(rates.mean()),
        "pooled_false_alarms_per_hour": sum(row["false_alarms"] for row in rows)
        / sum(row["hours"] for row in rows),
        "worst_subject_false_alarms_per_hour": float(rates.max()),
        "cvar80_subject_false_alarms_per_hour": external.empirical_cvar80(rates),
        "proxy_median_delay_seconds": float(np.median(delays)) if delays else math.nan,
        "proxy_p90_delay_seconds": float(np.percentile(delays, 90)) if delays else math.nan,
    }


def paired_bootstrap(
    external: Any,
    head_details: list[dict[str, Any]],
    zero_details: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    head = subject_contributions(head_details)
    zero = subject_contributions(zero_details)
    subjects = sorted(head)
    if subjects != sorted(zero):
        raise RuntimeError("OOF bootstrap subject mismatch")
    rng = np.random.default_rng(seed)
    differences = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(iterations):
        sampled = [subjects[int(index)] for index in rng.integers(0, len(subjects), len(subjects))]
        left = aggregate_subject_sample(external, head, sampled)
        right = aggregate_subject_sample(external, zero, sampled)
        for metric in BOOTSTRAP_METRICS:
            value = left[metric] - right[metric]
            if math.isfinite(value):
                differences[metric].append(value)
    output: dict[str, Any] = {}
    for metric, values in differences.items():
        array = np.asarray(values, dtype=np.float64)
        output[metric] = {
            "direction": "head_minus_zero",
            "valid_replicates": len(array),
            "mean_difference": float(array.mean()),
            "median_difference": float(np.median(array)),
            "ci95_lower": float(np.percentile(array, 2.5)),
            "ci95_upper": float(np.percentile(array, 97.5)),
            "positive": int(np.sum(array > 0)),
            "zero": int(np.sum(array == 0)),
            "negative": int(np.sum(array < 0)),
        }
    return output


def main() -> None:
    args = parse_args()
    for name in (
        "dataset_root",
        "external_code_dir",
        "gate2a_code_dir",
        "unimts_code_dir",
        "released_checkpoint",
        "weda_checkpoint",
        "fold_lock",
        "protocol",
        "cache_dir",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    for path in (args.released_checkpoint, args.weda_checkpoint, args.fold_lock, args.protocol):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.bootstrap_iterations < 1000:
        raise ValueError("bootstrap iterations must be at least 1000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    atomic_json(progress_path, {"state": "auditing"})
    external, weda_common = configure_imports(args)
    trials, dataset_audit = external.load_external_trials("smartfallmm", args.dataset_root)
    offsets, total_windows = attach_offsets(external, trials)
    folds, fold_payload = load_fold_lock(args.fold_lock, {trial.subject for trial in trials})
    fold_audit = []
    for index, subjects in enumerate(folds):
        selected = [trial for trial in trials if trial.subject in subjects]
        fold_audit.append(
            {
                "fold": index,
                "subjects": len(subjects),
                "young_subjects": sum(subject.startswith("young_") for subject in subjects),
                "old_subjects": sum(subject.startswith("old_") for subject in subjects),
                "fall_trials": sum(int(trial.is_fall) for trial in selected),
                "nonfall_trials": sum(int(not trial.is_fall) for trial in selected),
                "observable_falls": sum(
                    int(is_observable(external, trial)) for trial in selected if trial.is_fall
                ),
            }
        )
    audit = {
        "analysis_type": "version_locked_repeated_subject_level_cross_validation",
        "status": "internal_replication_not_independent_external_confirmation",
        "dataset_audit": dataset_audit,
        "fold_lock": fold_payload,
        "fold_audit": fold_audit,
        "total_windows": total_windows,
        "seeds": list(SEEDS),
        "primary_seed": SEEDS[0],
        "protocol_sha256": sha256(args.protocol),
    }
    atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps(clean_json({key: audit[key] for key in ("fold_audit", "total_windows", "seeds")}), indent=2), flush=True)
    if args.dry_run:
        print("SMARTFALLMM_HEAD_FIVEFOLD_DRY_RUN_OK", flush=True)
        return

    features, zero_scores, cache_manifest = create_or_load_cache(
        args,
        external,
        weda_common,
        trials,
        offsets,
        total_windows,
        dataset_audit["model_input_sha256"],
        progress_path,
    )
    initial_state = torch.load(cache_paths(args.cache_dir)["head"], map_location="cpu")
    fold_records: list[dict[str, Any]] = []
    zero_test_summaries: list[dict[str, Any]] = []
    primary_head_summaries: list[dict[str, Any]] = []
    oof_zero_details: list[dict[str, Any]] = []
    oof_head_details: list[dict[str, Any]] = []

    for fold_index in range(5):
        test_subjects = folds[fold_index]
        val_subjects = folds[(fold_index + 1) % 5]
        train_subjects = set().union(
            *[folds[index] for index in range(5) if index not in (fold_index, (fold_index + 1) % 5)]
        )
        val_trials, val_offsets, val_indices = subset(trials, offsets, val_subjects)
        test_trials, test_offsets, test_indices = subset(trials, offsets, test_subjects)
        val_labels = labels_for_trials(val_trials)
        zero_val_scores = np.asarray(zero_scores[val_indices], dtype=np.float32)
        zero_threshold, zero_val_summary, zero_eligible = select_threshold(
            external, val_trials, val_offsets, zero_val_scores
        )
        zero_test_scores = np.asarray(zero_scores[test_indices], dtype=np.float32)
        zero_test_summary, zero_details = evaluate(
            external, test_trials, test_offsets, zero_test_scores, zero_threshold
        )
        zero_test_summaries.append(zero_test_summary)
        oof_zero_details.extend(zero_details)
        for seed in SEEDS:
            atomic_json(
                progress_path,
                {"state": "training_head", "fold": fold_index, "seed": seed},
            )
            train_indices, train_labels, train_stats = training_indices(
                external,
                trials,
                offsets,
                train_subjects,
                args.negative_ratio,
                seed,
            )
            head, training = train_head(
                args,
                features,
                train_indices,
                train_labels,
                val_indices,
                val_labels,
                initial_state,
                seed,
            )
            head_val_scores = predict_head(head, features, val_indices, torch.device("cpu"))
            head_threshold, head_val_summary, head_eligible = select_threshold(
                external, val_trials, val_offsets, head_val_scores
            )
            head_test_scores = predict_head(head, features, test_indices, torch.device("cpu"))
            head_test_summary, head_details = evaluate(
                external, test_trials, test_offsets, head_test_scores, head_threshold
            )
            record = {
                "fold": fold_index,
                "seed": seed,
                "train_subjects": len(train_subjects),
                "validation_subjects": len(val_subjects),
                "test_subjects": len(test_subjects),
                "training_windows": train_stats,
                "training": training,
                "zero_shot": {
                    "selected_threshold": zero_threshold,
                    "eligible_thresholds": zero_eligible,
                    "validation": zero_val_summary,
                    "test": zero_test_summary,
                },
                "head_tune": {
                    "selected_threshold": head_threshold,
                    "eligible_thresholds": head_eligible,
                    "validation": head_val_summary,
                    "test": head_test_summary,
                },
            }
            fold_records.append(record)
            fold_dir = args.output_dir / f"fold{fold_index}" / f"seed{seed}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(fold_dir / "result.json", record)
            torch.save(
                {"head_state_dict": head.state_dict(), "fold": fold_index, "seed": seed},
                fold_dir / "best_head.pth",
            )
            if seed == SEEDS[0]:
                primary_head_summaries.append(head_test_summary)
                oof_head_details.extend(head_details)
            print(
                f"COMPLETE_FOLD_SEED fold={fold_index}/4 seed={seed} "
                f"zero_recall={zero_test_summary['observable_event_recall']:.6f} "
                f"head_recall={head_test_summary['observable_event_recall']:.6f} "
                f"zero_mean_fa_h={zero_test_summary['mean_subject_false_alarms_per_hour']:.6f} "
                f"head_mean_fa_h={head_test_summary['mean_subject_false_alarms_per_hour']:.6f}",
                flush=True,
            )
            del head

    primary_records = [row for row in fold_records if row["seed"] == SEEDS[0]]
    paired_fold_differences: dict[str, Any] = {}
    for metric in METRICS:
        values = np.asarray(
            [
                float(row["head_tune"]["test"][metric])
                - float(row["zero_shot"]["test"][metric])
                for row in primary_records
            ],
            dtype=np.float64,
        )
        paired_fold_differences[metric] = {
            "direction": "head_minus_zero",
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
    atomic_json(
        progress_path,
        {"state": "bootstrap", "iterations": args.bootstrap_iterations},
    )
    bootstrap = paired_bootstrap(
        external,
        oof_head_details,
        oof_zero_details,
        args.bootstrap_iterations,
        SEEDS[0],
    )
    zero_oof_contributions = subject_contributions(oof_zero_details)
    head_oof_contributions = subject_contributions(oof_head_details)
    oof_subjects = sorted(zero_oof_contributions)
    if oof_subjects != sorted(head_oof_contributions):
        raise RuntimeError("OOF point-estimate subject mismatch")
    zero_oof = aggregate_subject_sample(external, zero_oof_contributions, oof_subjects)
    head_oof = aggregate_subject_sample(external, head_oof_contributions, oof_subjects)
    all_head_summaries = [row["head_tune"]["test"] for row in fold_records]
    final = {
        "version": "smartfallmm_head_fivefold_v4_20260724",
        "status": "complete",
        "analysis_type": "version_locked_repeated_subject_level_cross_validation",
        "evidence_status": "internal_replication_not_independent_external_confirmation",
        "primary_seed": SEEDS[0],
        "sensitivity_seeds": list(SEEDS[1:]),
        "cache_manifest": cache_manifest,
        "zero_shot_fivefold": metric_summary(zero_test_summaries),
        "head_tune_primary_seed_fivefold": metric_summary(primary_head_summaries),
        "zero_shot_primary_oof": zero_oof,
        "head_tune_primary_oof": head_oof,
        "head_tune_all_fold_seed_runs": metric_summary(all_head_summaries),
        "paired_primary_fold_differences": paired_fold_differences,
        "paired_oof_subject_bootstrap": bootstrap,
        "fold_seed_records": fold_records,
        "required_next_gate": (
            "Apply this frozen protocol to UMAFall as an independent target-domain replication."
        ),
        "protocol_sha256": audit["protocol_sha256"],
        "fold_lock_sha256": sha256(args.fold_lock),
    }
    atomic_json(args.output_dir / "fivefold_results.json", final)
    atomic_json(progress_path, {"state": "complete"})
    print(json.dumps(clean_json({
        "zero_shot_fivefold": final["zero_shot_fivefold"],
        "head_tune_primary_seed_fivefold": final["head_tune_primary_seed_fivefold"],
        "zero_shot_primary_oof": final["zero_shot_primary_oof"],
        "head_tune_primary_oof": final["head_tune_primary_oof"],
        "paired_oof_subject_bootstrap": final["paired_oof_subject_bootstrap"],
    }), indent=2), flush=True)
    print("SMARTFALLMM_HEAD_FIVEFOLD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


TARGET_RECALL = 0.90
SEEDS = (20260724, 20260725, 20260726)
DURATION_METRICS = (
    "all_event_recall",
    "observable_event_recall",
    "mean_subject_false_alarms_per_hour",
    "pooled_false_alarms_per_hour",
    "worst_subject_false_alarms_per_hour",
    "cvar80_subject_false_alarms_per_hour",
    "mean_subject_false_alarm_duty_cycle",
    "pooled_false_alarm_duty_cycle",
    "worst_subject_false_alarm_duty_cycle",
    "cvar80_subject_false_alarm_duty_cycle",
    "mean_subject_total_alarm_duty_cycle",
    "pooled_total_alarm_duty_cycle",
    "event_external_alarm_seconds_per_hour",
    "median_alarm_segment_seconds",
    "p90_alarm_segment_seconds",
    "maximum_alarm_segment_seconds",
    "proxy_median_delay_seconds",
    "proxy_p90_delay_seconds",
)


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


def empirical_cvar80(values: np.ndarray) -> float:
    if not len(values):
        return math.nan
    threshold = float(np.quantile(values, 0.8))
    return float(values[values >= threshold].mean())


def new_head() -> nn.Module:
    return nn.Sequential(nn.LayerNorm(512), nn.Dropout(0.1), nn.Linear(512, 2))


def predict_head(
    head: nn.Module,
    features: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            values = np.asarray(features[indices[start : start + batch_size]], dtype=np.float32)
            outputs.append(torch.softmax(head(torch.from_numpy(values)), dim=1)[:, 1].numpy())
    return np.concatenate(outputs).astype(np.float32)


def duration_aware_alarm_segments(
    times: np.ndarray,
    scores: np.ndarray,
    tau_on: float,
    tau_off: float,
    k: int,
    refractory: float,
    monitoring_end: float,
) -> list[tuple[float, float]]:
    if len(times) != len(scores):
        raise ValueError("score time and score lengths differ")
    result: list[tuple[float, float]] = []
    active = False
    consecutive = 0
    segment_start = math.nan
    next_allowed = -math.inf
    for time_value, score_value in zip(times, scores):
        t = float(time_value)
        score = float(score_value)
        if active:
            if score <= tau_off:
                result.append((segment_start, t))
                active = False
                next_allowed = t + refractory
            continue
        if t < next_allowed:
            consecutive = 0
            continue
        consecutive = consecutive + 1 if score >= tau_on else 0
        if consecutive >= k:
            segment_start = t
            active = True
            consecutive = 0
    if active and len(times):
        result.append((segment_start, max(float(times[-1]), monitoring_end)))
    return result


def interval_overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def evaluate_trial_duration(
    external: Any,
    trial: Any,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    if trial.score_times is None or not len(trial.score_times):
        raise RuntimeError("score times were not prepared")
    params = external.Params(float(threshold), float(threshold), 1, 2.0)
    legacy = external.evaluate_trial(trial, scores, params)
    monitor_start = float(trial.score_times[0])
    monitor_end = max(monitor_start, float(trial.duration_seconds))
    monitoring = (monitor_start, monitor_end)
    segments = duration_aware_alarm_segments(
        trial.score_times,
        scores,
        params.tau_on,
        params.tau_off,
        params.k,
        params.refractory,
        monitor_end,
    )
    segments = [
        (max(start, monitor_start), min(end, monitor_end))
        for start, end in segments
        if min(end, monitor_end) > max(start, monitor_start)
    ]
    monitoring_seconds = max(0.0, monitor_end - monitor_start)
    if trial.is_fall:
        detection_end = min(
            float(trial.duration_seconds),
            float(trial.proxy_end) + float(external.GRACE_SECONDS),
        )
        event_interval = (float(trial.proxy_start), detection_end)
        event_monitoring_seconds = interval_overlap(monitoring, event_interval)
        event_alarm_seconds = sum(interval_overlap(segment, event_interval) for segment in segments)
    else:
        event_interval = None
        event_monitoring_seconds = 0.0
        event_alarm_seconds = 0.0
    durations = [max(0.0, end - start) for start, end in segments]
    alarm_seconds = float(sum(durations))
    false_alarm_seconds = max(0.0, alarm_seconds - event_alarm_seconds)
    non_event_monitoring_seconds = max(0.0, monitoring_seconds - event_monitoring_seconds)
    output = dict(legacy)
    output.update(
        {
            "observable": int(
                bool(trial.is_fall)
                and monitor_start
                <= min(
                    float(trial.duration_seconds),
                    float(trial.proxy_end) + float(external.GRACE_SECONDS),
                )
            ),
            "monitoring_start_seconds": monitor_start,
            "monitoring_seconds": monitoring_seconds,
            "non_event_monitoring_seconds": non_event_monitoring_seconds,
            "alarm_seconds": alarm_seconds,
            "event_alarm_seconds": event_alarm_seconds,
            "event_external_alarm_seconds": false_alarm_seconds,
            "total_alarm_duty_cycle": alarm_seconds / monitoring_seconds
            if monitoring_seconds > 0
            else math.nan,
            "false_alarm_duty_cycle": false_alarm_seconds / non_event_monitoring_seconds
            if non_event_monitoring_seconds > 0
            else math.nan,
            "alarm_segment_durations_seconds": durations,
            "maximum_alarm_segment_seconds": max(durations, default=0.0),
            "duration_aware_alarm_segments": segments,
            "duration_event_interval": event_interval,
        }
    )
    return output


def subject_contributions(details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        groups[str(row["subject"])].append(row)
    output: dict[str, dict[str, Any]] = {}
    for subject, rows in groups.items():
        falls = [row for row in rows if int(row["is_fall"]) == 1]
        observable = [row for row in falls if int(row["observable"]) == 1]
        output[subject] = {
            "false_alarms": sum(float(row["false_alarms"]) for row in rows),
            "legacy_non_event_hours": sum(float(row["non_event_hours"]) for row in rows),
            "monitoring_seconds": sum(float(row["monitoring_seconds"]) for row in rows),
            "non_event_monitoring_seconds": sum(
                float(row["non_event_monitoring_seconds"]) for row in rows
            ),
            "alarm_seconds": sum(float(row["alarm_seconds"]) for row in rows),
            "false_alarm_seconds": sum(
                float(row["event_external_alarm_seconds"]) for row in rows
            ),
            "all_falls": len(falls),
            "detected_all": sum(int(row["detected"]) for row in falls),
            "observable_falls": len(observable),
            "detected_observable": sum(int(row["detected"]) for row in observable),
            "delays": [
                float(row["proxy_delay_seconds"])
                for row in falls
                if row["proxy_delay_seconds"] is not None
            ],
            "segment_durations": [
                float(value)
                for row in rows
                for value in row["alarm_segment_durations_seconds"]
            ],
        }
    return output


def aggregate_subject_sample(
    contributions: dict[str, dict[str, Any]], sampled: list[str]
) -> dict[str, float]:
    rows = [contributions[subject] for subject in sampled]
    valid_fa_rows = [row for row in rows if row["legacy_non_event_hours"] > 0]
    valid_duty_rows = [row for row in rows if row["non_event_monitoring_seconds"] > 0]
    valid_total_rows = [row for row in rows if row["monitoring_seconds"] > 0]
    fa_rates = np.asarray(
        [row["false_alarms"] / row["legacy_non_event_hours"] for row in valid_fa_rows],
        dtype=np.float64,
    )
    false_duties = np.asarray(
        [row["false_alarm_seconds"] / row["non_event_monitoring_seconds"] for row in valid_duty_rows],
        dtype=np.float64,
    )
    total_duties = np.asarray(
        [row["alarm_seconds"] / row["monitoring_seconds"] for row in valid_total_rows],
        dtype=np.float64,
    )
    all_falls = sum(row["all_falls"] for row in rows)
    observable = sum(row["observable_falls"] for row in rows)
    delays = [value for row in rows for value in row["delays"]]
    durations = [value for row in rows for value in row["segment_durations"]]
    total_legacy_hours = sum(row["legacy_non_event_hours"] for row in rows)
    total_non_event_seconds = sum(row["non_event_monitoring_seconds"] for row in rows)
    total_monitoring_seconds = sum(row["monitoring_seconds"] for row in rows)
    total_false_seconds = sum(row["false_alarm_seconds"] for row in rows)
    return {
        "all_event_recall": sum(row["detected_all"] for row in rows) / all_falls
        if all_falls
        else math.nan,
        "observable_event_recall": sum(row["detected_observable"] for row in rows) / observable
        if observable
        else math.nan,
        "observable_coverage": observable / all_falls if all_falls else math.nan,
        "mean_subject_false_alarms_per_hour": float(fa_rates.mean()),
        "pooled_false_alarms_per_hour": sum(row["false_alarms"] for row in rows)
        / total_legacy_hours,
        "worst_subject_false_alarms_per_hour": float(fa_rates.max()),
        "cvar80_subject_false_alarms_per_hour": empirical_cvar80(fa_rates),
        "mean_subject_false_alarm_duty_cycle": float(false_duties.mean()),
        "pooled_false_alarm_duty_cycle": total_false_seconds / total_non_event_seconds,
        "worst_subject_false_alarm_duty_cycle": float(false_duties.max()),
        "cvar80_subject_false_alarm_duty_cycle": empirical_cvar80(false_duties),
        "mean_subject_total_alarm_duty_cycle": float(total_duties.mean()),
        "pooled_total_alarm_duty_cycle": sum(row["alarm_seconds"] for row in rows)
        / total_monitoring_seconds,
        "event_external_alarm_seconds_per_hour": total_false_seconds
        / (total_non_event_seconds / 3600.0),
        "median_alarm_segment_seconds": float(np.median(durations)) if durations else 0.0,
        "p90_alarm_segment_seconds": float(np.percentile(durations, 90)) if durations else 0.0,
        "maximum_alarm_segment_seconds": max(durations, default=0.0),
        "proxy_median_delay_seconds": float(np.median(delays)) if delays else math.nan,
        "proxy_p90_delay_seconds": float(np.percentile(delays, 90)) if delays else math.nan,
        "all_fall_events": all_falls,
        "observable_fall_events": observable,
        "subjects": len(rows),
        "alarm_seconds": sum(row["alarm_seconds"] for row in rows),
        "event_external_alarm_seconds": total_false_seconds,
        "monitoring_seconds": total_monitoring_seconds,
        "non_event_monitoring_seconds": total_non_event_seconds,
    }


def evaluate_duration(
    external: Any,
    trials: list[Any],
    offsets: list[tuple[int, int]],
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = [
        evaluate_trial_duration(external, trial, scores[start:end], threshold)
        for trial, (start, end) in zip(trials, offsets)
    ]
    contributions = subject_contributions(details)
    summary = aggregate_subject_sample(contributions, sorted(contributions))
    summary["threshold"] = float(threshold)
    return summary, details


def threshold_sort_key(summary: dict[str, Any], threshold: float) -> tuple[float, ...]:
    return (
        float(summary["mean_subject_false_alarm_duty_cycle"]),
        float(summary["pooled_false_alarm_duty_cycle"]),
        float(summary["mean_subject_false_alarms_per_hour"]),
        float(summary["cvar80_subject_false_alarm_duty_cycle"]),
        float(summary["worst_subject_false_alarm_duty_cycle"]),
        float(summary["proxy_median_delay_seconds"]),
        -float(threshold),
    )


def select_duration_threshold(
    external: Any,
    trials: list[Any],
    offsets: list[tuple[int, int]],
    scores: np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> tuple[float, dict[str, Any], int]:
    evaluated = [
        (float(threshold), evaluate_duration(external, trials, offsets, scores, threshold)[0])
        for threshold in np.linspace(0.01, 0.99, 99)
    ]
    eligible = [
        row for row in evaluated if row[1]["observable_event_recall"] >= target_recall
    ]
    if not eligible:
        best = max(row[1]["observable_event_recall"] for row in evaluated)
        raise RuntimeError(
            f"no duration-aware threshold reaches target observable recall; best={best}"
        )
    selected = min(eligible, key=lambda row: threshold_sort_key(row[1], row[0]))
    return selected[0], selected[1], len(eligible)


def paired_bootstrap(
    left_details: list[dict[str, Any]],
    right_details: list[dict[str, Any]],
    iterations: int,
    seed: int,
    direction: str,
) -> dict[str, Any]:
    left = subject_contributions(left_details)
    right = subject_contributions(right_details)
    subjects = sorted(left)
    if subjects != sorted(right):
        raise RuntimeError("paired bootstrap subject mismatch")
    rng = np.random.default_rng(seed)
    differences = {metric: [] for metric in DURATION_METRICS}
    for _ in range(iterations):
        sampled = [subjects[int(index)] for index in rng.integers(0, len(subjects), len(subjects))]
        left_row = aggregate_subject_sample(left, sampled)
        right_row = aggregate_subject_sample(right, sampled)
        for metric in DURATION_METRICS:
            value = left_row[metric] - right_row[metric]
            if math.isfinite(value):
                differences[metric].append(value)
    output: dict[str, Any] = {}
    for metric, values in differences.items():
        array = np.asarray(values, dtype=np.float64)
        output[metric] = {
            "direction": direction,
            "valid_replicates": len(array),
            "mean_difference": float(array.mean()),
            "ci95_lower": float(np.percentile(array, 2.5)),
            "ci95_upper": float(np.percentile(array, 97.5)),
            "positive": int(np.sum(array > 0)),
            "zero": int(np.sum(array == 0)),
            "negative": int(np.sum(array < 0)),
        }
    return output


@dataclass(frozen=True)
class LossProfile:
    variant: str
    alpha_duty: float
    beta_cvar: float
    gamma_temporal: float

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.variant}_a{self.alpha_duty:g}_b{self.beta_cvar:g}_"
            f"g{self.gamma_temporal:g}"
        ).replace(".", "p")


def candidate_grid(variant: str) -> list[LossProfile]:
    if variant == "event_mil":
        return [LossProfile(variant, 0.0, 0.0, 0.0)]
    if variant == "event_duty":
        return [LossProfile(variant, alpha, 0.0, 0.0) for alpha in (0.5, 1.0)]
    if variant == "event_duty_cvar":
        return [
            LossProfile(variant, alpha, beta, 0.0)
            for alpha in (0.5, 1.0)
            for beta in (0.5, 1.0)
        ]
    if variant == "esra_full":
        return [
            LossProfile(variant, alpha, beta, gamma)
            for alpha in (0.5, 1.0)
            for beta in (0.5, 1.0)
            for gamma in (0.01, 0.05)
        ]
    raise ValueError(f"unknown variant: {variant}")


@dataclass
class TrainingPool:
    positive_bags: list[np.ndarray]
    negative_bags: list[np.ndarray]
    temporal_bags: list[np.ndarray]
    background_by_subject: dict[str, np.ndarray]


def build_training_pool(
    external: Any,
    trials: list[Any],
    offsets: list[tuple[int, int]],
    train_subjects: set[str],
    observable_fn: Callable[[Any, Any], bool],
) -> TrainingPool:
    positive_bags: list[np.ndarray] = []
    negative_bags: list[np.ndarray] = []
    temporal_bags: list[np.ndarray] = []
    background: dict[str, list[int]] = defaultdict(list)
    for trial, (start, end) in zip(trials, offsets):
        if str(trial.subject) not in train_subjects:
            continue
        indices = np.arange(start, end, dtype=np.int64)
        temporal_bags.append(indices)
        if trial.is_fall:
            positive_mask = (
                (trial.score_times >= float(trial.proxy_start))
                & (trial.score_times <= float(trial.proxy_end))
            )
            if observable_fn(external, trial) and np.any(positive_mask):
                positive_bags.append(indices[positive_mask])
            detection_end = min(
                float(trial.duration_seconds),
                float(trial.proxy_end) + float(external.GRACE_SECONDS),
            )
            background_mask = (trial.score_times < float(trial.proxy_start)) | (
                trial.score_times > detection_end
            )
            background[str(trial.subject)].extend(indices[background_mask].tolist())
        else:
            negative_bags.append(indices)
            background[str(trial.subject)].extend(indices.tolist())
    if not positive_bags or not negative_bags or not background:
        raise RuntimeError("ESRA training needs positive, negative, and background examples")
    return TrainingPool(
        positive_bags=positive_bags,
        negative_bags=negative_bags,
        temporal_bags=temporal_bags,
        background_by_subject={
            subject: np.asarray(values, dtype=np.int64) for subject, values in background.items()
        },
    )


def _probabilities(head: nn.Module, features: np.ndarray, indices: np.ndarray) -> torch.Tensor:
    values = np.asarray(features[indices], dtype=np.float32)
    return torch.softmax(head(torch.from_numpy(values)), dim=1)[:, 1]


def _soft_max(values: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    weights = torch.softmax(values / temperature, dim=0)
    return torch.sum(weights * values)


def _sample_rows(rng: np.random.Generator, rows: list[np.ndarray], count: int) -> list[np.ndarray]:
    indices = rng.integers(0, len(rows), size=count)
    return [rows[int(index)] for index in indices]


def compute_esra_loss(
    head: nn.Module,
    features: np.ndarray,
    pool: TrainingPool,
    profile: LossProfile,
    rng: np.random.Generator,
    positive_bags_per_step: int,
    negative_bags_per_step: int,
    subjects_per_step: int,
    background_windows_per_subject: int,
    temporal_bags_per_step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    epsilon = 1e-6
    positive_scores = [
        _soft_max(_probabilities(head, features, bag))
        for bag in _sample_rows(rng, pool.positive_bags, positive_bags_per_step)
    ]
    negative_scores = [
        _soft_max(_probabilities(head, features, bag))
        for bag in _sample_rows(rng, pool.negative_bags, negative_bags_per_step)
    ]
    positive_loss = torch.stack([-torch.log(score.clamp(epsilon, 1.0 - epsilon)) for score in positive_scores]).mean()
    negative_loss = torch.stack([-torch.log((1.0 - score).clamp(epsilon, 1.0 - epsilon)) for score in negative_scores]).mean()
    event_loss = 0.5 * (positive_loss + negative_loss)

    subjects = sorted(pool.background_by_subject)
    selected_count = min(subjects_per_step, len(subjects))
    selected_subjects = rng.choice(subjects, size=selected_count, replace=False).tolist()
    burdens: list[torch.Tensor] = []
    for subject in selected_subjects:
        available = pool.background_by_subject[str(subject)]
        if len(available) > background_windows_per_subject:
            chosen = rng.choice(available, size=background_windows_per_subject, replace=False)
        else:
            chosen = available
        burdens.append(_probabilities(head, features, np.asarray(chosen, dtype=np.int64)).mean())
    burden_tensor = torch.stack(burdens)
    duty_loss = burden_tensor.mean()
    tail_count = max(1, math.ceil(0.2 * len(burden_tensor)))
    cvar_loss = torch.topk(burden_tensor, k=tail_count).values.mean()

    temporal_values: list[torch.Tensor] = []
    for bag in _sample_rows(rng, pool.temporal_bags, temporal_bags_per_step):
        probabilities = _probabilities(head, features, bag)
        if len(probabilities) > 1:
            temporal_values.append(torch.abs(probabilities[1:] - probabilities[:-1]).mean())
    temporal_loss = (
        torch.stack(temporal_values).mean()
        if temporal_values
        else torch.tensor(0.0, dtype=torch.float32)
    )
    total = (
        event_loss
        + profile.alpha_duty * duty_loss
        + profile.beta_cvar * cvar_loss
        + profile.gamma_temporal * temporal_loss
    )
    return total, {
        "event": float(event_loss.detach()),
        "duty": float(duty_loss.detach()),
        "cvar": float(cvar_loss.detach()),
        "temporal": float(temporal_loss.detach()),
        "total": float(total.detach()),
    }


def validation_event_ap(
    head: nn.Module,
    features: np.ndarray,
    trials: list[Any],
    local_offsets: list[tuple[int, int]],
    global_indices: np.ndarray,
    observable_fn: Callable[[Any, Any], bool],
    external: Any,
) -> float:
    labels: list[int] = []
    values: list[float] = []
    all_scores = predict_head(head, features, global_indices)
    for trial, (start, end) in zip(trials, local_offsets):
        scores = all_scores[start:end]
        if trial.is_fall:
            if not observable_fn(external, trial):
                continue
            mask = (
                (trial.score_times >= float(trial.proxy_start))
                & (trial.score_times <= float(trial.proxy_end))
            )
            if not np.any(mask):
                continue
            scores = scores[mask]
            labels.append(1)
        else:
            labels.append(0)
        weights = np.exp((scores - float(np.max(scores))) / 0.10)
        values.append(float(np.sum(weights * scores) / np.sum(weights)))
    return float(average_precision_score(np.asarray(labels), np.asarray(values)))


def train_esra_candidate(
    features: np.ndarray,
    pool: TrainingPool,
    val_trials: list[Any],
    val_offsets: list[tuple[int, int]],
    val_indices: np.ndarray,
    external: Any,
    observable_fn: Callable[[Any, Any], bool],
    initial_state: dict[str, Any],
    profile: LossProfile,
    seed: int,
    epochs: int,
    patience: int,
    steps_per_epoch: int,
    learning_rate: float,
    weight_decay: float,
    positive_bags_per_step: int,
    negative_bags_per_step: int,
    subjects_per_step: int,
    background_windows_per_subject: int,
    temporal_bags_per_step: int,
) -> tuple[nn.Module, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    head = new_head()
    head.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.05
    )
    best_ap = -math.inf
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        head.train()
        component_totals: dict[str, float] = defaultdict(float)
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            loss, components = compute_esra_loss(
                head,
                features,
                pool,
                profile,
                rng,
                positive_bags_per_step,
                negative_bags_per_step,
                subjects_per_step,
                background_windows_per_subject,
                temporal_bags_per_step,
            )
            loss.backward()
            optimizer.step()
            for key, value in components.items():
                component_totals[key] += value
        scheduler.step()
        event_ap = validation_event_ap(
            head,
            features,
            val_trials,
            val_offsets,
            val_indices,
            observable_fn,
            external,
        )
        history.append(
            {
                "epoch": epoch,
                "validation_event_average_precision": event_ap,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "loss_components": {
                    key: value / steps_per_epoch for key, value in component_totals.items()
                },
            }
        )
        if event_ap > best_ap + 1e-8:
            best_ap = event_ap
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("ESRA training produced no checkpoint")
    head.load_state_dict(best_state, strict=True)
    return head, {
        "seed": seed,
        "profile": asdict(profile),
        "candidate_id": profile.candidate_id,
        "best_epoch": best_epoch,
        "best_validation_event_average_precision": best_ap,
        "epochs_run": len(history),
        "steps_per_epoch": steps_per_epoch,
        "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "history": history,
    }

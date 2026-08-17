from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


C_GRID = (0.01, 0.1, 1.0, 10.0)


class LinearEventHead(nn.Module):
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.linear = nn.Linear(int(len(mean)), 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = (values - self.feature_mean) / self.feature_scale
        return torch.sigmoid(self.linear(normalized)).squeeze(-1)


def head_from_pipeline(model: Pipeline) -> LinearEventHead:
    scaler = model.named_steps["scale"]
    classifier = model.named_steps["classifier"]
    head = LinearEventHead(scaler.mean_, scaler.scale_)
    with torch.no_grad():
        head.linear.weight.copy_(
            torch.as_tensor(classifier.coef_, dtype=torch.float32)
        )
        head.linear.bias.copy_(
            torch.as_tensor(classifier.intercept_, dtype=torch.float32)
        )
    return head


def predict(
    head: LinearEventHead,
    features: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    head.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = np.asarray(
                features[indices[start : start + batch_size]], dtype=np.float32
            )
            output.append(head(torch.from_numpy(batch)).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def all_training_indices(
    external: Any,
    trials: list[Any],
    offsets: list[tuple[int, int]],
    subjects: set[str],
    observable_fn: Callable[[Any, Any], bool],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    positive: list[int] = []
    negative: list[int] = []
    excluded = 0
    for trial, (start, end) in zip(trials, offsets):
        if str(trial.subject) not in subjects:
            continue
        indices = np.arange(start, end, dtype=np.int64)
        if trial.is_fall:
            if not observable_fn(external, trial):
                excluded += 1
                continue
            mask = (
                (trial.score_times >= float(trial.proxy_start))
                & (trial.score_times <= float(trial.proxy_end))
            )
            positive.extend(indices[mask].tolist())
        else:
            negative.extend(indices.tolist())
    if not positive or not negative:
        raise RuntimeError("linear probe training requires both classes")
    indices = np.asarray(positive + negative, dtype=np.int64)
    labels = np.asarray(
        [1] * len(positive) + [0] * len(negative), dtype=np.int64
    )
    return indices, labels, {
        "positive": len(positive),
        "negative": len(negative),
        "total": len(indices),
        "excluded_unobservable_fall_trials": excluded,
    }


def fit_probe_candidates(
    v6_core: Any,
    external: Any,
    features: np.ndarray,
    train_indices: np.ndarray,
    train_labels: np.ndarray,
    val_trials: list[Any],
    val_offsets: list[tuple[int, int]],
    val_indices: np.ndarray,
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]]]:
    train_values = np.asarray(features[train_indices], dtype=np.float32)
    val_values = np.asarray(features[val_indices], dtype=np.float32)
    candidates: list[dict[str, Any]] = []
    models: dict[float, Pipeline] = {}
    for order, c_value in enumerate(C_GRID):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=3000,
                        solver="lbfgs",
                        random_state=20260724,
                    ),
                ),
            ]
        )
        model.fit(train_values, train_labels)
        scores = model.predict_proba(val_values)[:, 1].astype(np.float32)
        try:
            threshold, validation, eligible = v6_core.select_duration_threshold(
                external, val_trials, val_offsets, scores
            )
            record = {
                "candidate_id": f"linear_probe_C{c_value:g}",
                "candidate_order": order,
                "C": c_value,
                "status": "eligible",
                "selected_threshold": threshold,
                "eligible_thresholds": eligible,
                "validation": validation,
            }
        except RuntimeError as error:
            record = {
                "candidate_id": f"linear_probe_C{c_value:g}",
                "candidate_order": order,
                "C": c_value,
                "status": "ineligible",
                "error": str(error),
            }
        candidates.append(record)
        models[c_value] = model
    eligible = [row for row in candidates if row["status"] == "eligible"]
    if not eligible:
        raise RuntimeError("no linear probe candidate satisfies validation recall")
    selected = min(
        eligible,
        key=lambda row: v6_core.threshold_sort_key(
            row["validation"], float(row["selected_threshold"])
        )
        + (float(row["candidate_order"]),),
    )
    return models[float(selected["C"])], selected, candidates


@dataclass(frozen=True)
class Profile:
    variant: str
    alpha_duty: float
    lambda_anchor: float = 0.01

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.variant}_a{self.alpha_duty:g}_l{self.lambda_anchor:g}"
        ).replace(".", "p")


def candidate_grid(variant: str) -> list[Profile]:
    if variant == "linear_event_mil":
        return [Profile(variant, 0.0, 0.01)]
    if variant == "linear_event_duty":
        return [Profile(variant, alpha, 0.01) for alpha in (0.5, 1.0)]
    raise ValueError(variant)


def _sample_rows(
    rng: np.random.Generator, rows: list[np.ndarray], count: int
) -> list[np.ndarray]:
    selected = rng.integers(0, len(rows), size=count)
    return [rows[int(index)] for index in selected]


def _probabilities(
    head: LinearEventHead, features: np.ndarray, indices: np.ndarray
) -> torch.Tensor:
    values = np.asarray(features[indices], dtype=np.float32)
    return head(torch.from_numpy(values))


def _soft_max(values: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    weights = torch.softmax(values / temperature, dim=0)
    return torch.sum(weights * values)


def compute_loss(
    head: LinearEventHead,
    features: np.ndarray,
    pool: Any,
    profile: Profile,
    initial_weight: torch.Tensor,
    initial_bias: torch.Tensor,
    rng: np.random.Generator,
    positive_bags_per_step: int,
    negative_bags_per_step: int,
    subjects_per_step: int,
    background_windows_per_subject: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    epsilon = 1e-6
    positive = [
        _soft_max(_probabilities(head, features, bag))
        for bag in _sample_rows(rng, pool.positive_bags, positive_bags_per_step)
    ]
    negative = [
        _soft_max(_probabilities(head, features, bag))
        for bag in _sample_rows(rng, pool.negative_bags, negative_bags_per_step)
    ]
    positive_loss = torch.stack(
        [-torch.log(value.clamp(epsilon, 1.0 - epsilon)) for value in positive]
    ).mean()
    negative_loss = torch.stack(
        [-torch.log((1.0 - value).clamp(epsilon, 1.0 - epsilon)) for value in negative]
    ).mean()
    event_loss = 0.5 * (positive_loss + negative_loss)

    subjects = sorted(pool.background_by_subject)
    selected_count = min(subjects_per_step, len(subjects))
    chosen_subjects = rng.choice(
        subjects, size=selected_count, replace=False
    ).tolist()
    burdens: list[torch.Tensor] = []
    for subject in chosen_subjects:
        available = pool.background_by_subject[str(subject)]
        if len(available) > background_windows_per_subject:
            chosen = rng.choice(
                available, size=background_windows_per_subject, replace=False
            )
        else:
            chosen = available
        burdens.append(
            _probabilities(
                head, features, np.asarray(chosen, dtype=np.int64)
            ).mean()
        )
    duty_loss = torch.stack(burdens).mean()
    anchor_loss = torch.mean((head.linear.weight - initial_weight) ** 2)
    anchor_loss = anchor_loss + torch.mean(
        (head.linear.bias - initial_bias) ** 2
    )
    total = (
        event_loss
        + profile.alpha_duty * duty_loss
        + profile.lambda_anchor * anchor_loss
    )
    return total, {
        "event": float(event_loss.detach()),
        "duty": float(duty_loss.detach()),
        "anchor": float(anchor_loss.detach()),
        "total": float(total.detach()),
    }


def validation_event_ap(
    head: LinearEventHead,
    features: np.ndarray,
    trials: list[Any],
    offsets: list[tuple[int, int]],
    indices: np.ndarray,
    external: Any,
    observable_fn: Callable[[Any, Any], bool],
) -> float:
    labels: list[int] = []
    values: list[float] = []
    scores = predict(head, features, indices)
    for trial, (start, end) in zip(trials, offsets):
        trial_scores = scores[start:end]
        if trial.is_fall:
            if not observable_fn(external, trial):
                continue
            mask = (
                (trial.score_times >= float(trial.proxy_start))
                & (trial.score_times <= float(trial.proxy_end))
            )
            if not np.any(mask):
                continue
            trial_scores = trial_scores[mask]
            labels.append(1)
        else:
            labels.append(0)
        weights = np.exp(
            (trial_scores - float(np.max(trial_scores))) / 0.10
        )
        values.append(
            float(np.sum(weights * trial_scores) / np.sum(weights))
        )
    return float(
        average_precision_score(np.asarray(labels), np.asarray(values))
    )


def train_candidate(
    v6_core: Any,
    features: np.ndarray,
    pool: Any,
    val_trials: list[Any],
    val_offsets: list[tuple[int, int]],
    val_indices: np.ndarray,
    external: Any,
    observable_fn: Callable[[Any, Any], bool],
    probe: Pipeline,
    profile: Profile,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    learning_rate: float,
    weight_decay: float,
    positive_bags_per_step: int,
    negative_bags_per_step: int,
    subjects_per_step: int,
    background_windows_per_subject: int,
) -> tuple[LinearEventHead, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    head = head_from_pipeline(probe)
    initial_weight = head.linear.weight.detach().clone()
    initial_bias = head.linear.bias.detach().clone()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.05
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_threshold = 0.0
    best_validation: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        head.train()
        totals: dict[str, float] = defaultdict(float)
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            loss, components = compute_loss(
                head,
                features,
                pool,
                profile,
                initial_weight,
                initial_bias,
                rng,
                positive_bags_per_step,
                negative_bags_per_step,
                subjects_per_step,
                background_windows_per_subject,
            )
            loss.backward()
            optimizer.step()
            for key, value in components.items():
                totals[key] += value
        scheduler.step()
        scores = predict(head, features, val_indices)
        event_ap = validation_event_ap(
            head,
            features,
            val_trials,
            val_offsets,
            val_indices,
            external,
            observable_fn,
        )
        try:
            threshold, validation, eligible = v6_core.select_duration_threshold(
                external, val_trials, val_offsets, scores
            )
            status = "eligible"
            key = v6_core.threshold_sort_key(validation, threshold)
            if best_key is None or key < best_key:
                best_key = key
                best_state = copy.deepcopy(head.state_dict())
                best_epoch = epoch
                best_threshold = float(threshold)
                best_validation = validation
        except RuntimeError as error:
            threshold = None
            validation = None
            eligible = 0
            status = f"ineligible: {error}"
        history.append(
            {
                "epoch": epoch,
                "status": status,
                "validation_event_average_precision": event_ap,
                "selected_threshold": threshold,
                "eligible_thresholds": eligible,
                "validation": validation,
                "loss_components": {
                    key: value / steps_per_epoch
                    for key, value in totals.items()
                },
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
    if best_state is None or best_validation is None:
        raise RuntimeError(
            f"candidate {profile.candidate_id} never reached validation recall"
        )
    head.load_state_dict(best_state, strict=True)
    return head, {
        "profile": asdict(profile),
        "candidate_id": profile.candidate_id,
        "seed": seed,
        "trainable_parameters": sum(
            parameter.numel() for parameter in head.parameters()
        ),
        "best_epoch": best_epoch,
        "selected_threshold": best_threshold,
        "validation": best_validation,
        "epochs_run": epochs,
        "history": history,
    }

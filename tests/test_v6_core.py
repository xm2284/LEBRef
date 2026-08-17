from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

import esra_core as target


@dataclass(frozen=True)
class Params:
    tau_on: float
    tau_off: float
    k: int
    refractory: float


@dataclass
class Trial:
    subject: str
    is_fall: bool
    score_times: np.ndarray
    duration_seconds: float
    proxy_start: float | None = None
    proxy_end: float | None = None


class External:
    GRACE_SECONDS = 0.0
    Params = Params

    @staticmethod
    def evaluate_trial(trial: Trial, scores: np.ndarray, params: Params):
        segments = target.duration_aware_alarm_segments(
            trial.score_times,
            scores,
            params.tau_on,
            params.tau_off,
            params.k,
            params.refractory,
            float(trial.score_times[-1]),
        )
        if trial.is_fall:
            event = (float(trial.proxy_start), float(trial.proxy_end))
            detected_segments = [s for s in segments if target.interval_overlap(s, event) > 0]
            false_segments = [s for s in segments if s not in detected_segments]
            detected = int(bool(detected_segments))
            delay = (
                max(detected_segments[0][0], event[0]) - event[0]
                if detected_segments
                else None
            )
            event_exposure = event[1] - event[0]
        else:
            false_segments = segments
            detected = 0
            delay = None
            event_exposure = 0.0
        return {
            "subject": trial.subject,
            "scene": "synthetic",
            "trial_id": f"{trial.subject}-{int(trial.is_fall)}",
            "is_fall": int(trial.is_fall),
            "detected": detected,
            "proxy_delay_seconds": delay,
            "false_alarms": len(false_segments),
            "non_event_hours": (trial.duration_seconds - event_exposure) / 3600.0,
        }


def make_trial(subject: str, is_fall: bool) -> Trial:
    return Trial(
        subject=subject,
        is_fall=is_fall,
        score_times=np.asarray([4.0, 5.0, 6.0, 7.0, 8.0]),
        duration_seconds=9.0,
        proxy_start=5.0 if is_fall else None,
        proxy_end=6.0 if is_fall else None,
    )


def test_head_parameter_count() -> None:
    assert sum(parameter.numel() for parameter in target.new_head().parameters()) == 2050


def test_long_alarm_is_split_outside_event() -> None:
    trial = make_trial("s1", True)
    row = target.evaluate_trial_duration(External, trial, np.ones(5), 0.5)
    assert row["detected"] == 1
    assert row["false_alarms"] == 0
    assert math.isclose(row["alarm_seconds"], 5.0)
    assert math.isclose(row["event_alarm_seconds"], 1.0)
    assert math.isclose(row["event_external_alarm_seconds"], 4.0)
    assert math.isclose(row["false_alarm_duty_cycle"], 1.0)


def test_always_on_nonfall_has_full_duty() -> None:
    trial = make_trial("s2", False)
    row = target.evaluate_trial_duration(External, trial, np.ones(5), 0.5)
    assert row["false_alarms"] == 1
    assert math.isclose(row["total_alarm_duty_cycle"], 1.0)
    assert math.isclose(row["false_alarm_duty_cycle"], 1.0)


def test_duration_threshold_avoids_always_on_solution() -> None:
    trials = [make_trial("positive", True), make_trial("negative", False)]
    offsets = [(0, 5), (5, 10)]
    scores = np.asarray([0.1, 0.8, 0.8, 0.1, 0.1, 0.6, 0.6, 0.6, 0.6, 0.6])
    threshold, summary, eligible = target.select_duration_threshold(
        External, trials, offsets, scores
    )
    assert math.isclose(threshold, 0.80)
    assert summary["observable_event_recall"] == 1.0
    assert summary["pooled_false_alarm_duty_cycle"] == 0.0
    assert eligible > 0


def test_candidate_grid_is_locked() -> None:
    assert len(target.candidate_grid("event_mil")) == 1
    assert len(target.candidate_grid("event_duty")) == 2
    assert len(target.candidate_grid("event_duty_cvar")) == 4
    assert len(target.candidate_grid("esra_full")) == 8


def test_esra_loss_is_finite_and_differentiable() -> None:
    features = np.random.default_rng(7).normal(size=(24, 512)).astype(np.float32)
    pool = target.TrainingPool(
        positive_bags=[np.asarray([0, 1, 2])],
        negative_bags=[np.asarray([3, 4, 5, 6])],
        temporal_bags=[np.asarray([7, 8, 9, 10])],
        background_by_subject={
            "s1": np.asarray([11, 12, 13, 14]),
            "s2": np.asarray([15, 16, 17, 18]),
            "s3": np.asarray([19, 20, 21, 22, 23]),
        },
    )
    head = target.new_head()
    loss, components = target.compute_esra_loss(
        head,
        features,
        pool,
        target.LossProfile("esra_full", 1.0, 1.0, 0.05),
        np.random.default_rng(11),
        2,
        2,
        3,
        4,
        2,
    )
    assert torch.isfinite(loss)
    assert set(components) == {"event", "duty", "cvar", "temporal", "total"}
    loss.backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_one_epoch_candidate_training_smoke() -> None:
    features = np.random.default_rng(17).normal(size=(40, 512)).astype(np.float32)
    pool = target.TrainingPool(
        positive_bags=[np.asarray([0, 1, 2])],
        negative_bags=[np.asarray([3, 4, 5, 6])],
        temporal_bags=[np.asarray([7, 8, 9, 10])],
        background_by_subject={
            "s1": np.asarray([11, 12, 13, 14]),
            "s2": np.asarray([15, 16, 17, 18]),
        },
    )
    val_trials = [make_trial("vp", True), make_trial("vn", False)]
    val_offsets = [(0, 5), (5, 10)]
    val_indices = np.arange(24, 34, dtype=np.int64)
    initial_state = target.new_head().state_dict()
    head, training = target.train_esra_candidate(
        features,
        pool,
        val_trials,
        val_offsets,
        val_indices,
        External,
        lambda external, trial: bool(trial.is_fall),
        initial_state,
        target.LossProfile("event_mil", 0.0, 0.0, 0.0),
        seed=19,
        epochs=1,
        patience=1,
        steps_per_epoch=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        positive_bags_per_step=1,
        negative_bags_per_step=1,
        subjects_per_step=2,
        background_windows_per_subject=4,
        temporal_bags_per_step=1,
    )
    assert sum(parameter.numel() for parameter in head.parameters()) == 2050
    assert training["epochs_run"] == 1
    assert math.isfinite(training["best_validation_event_average_precision"])


def main() -> None:
    test_head_parameter_count()
    test_long_alarm_is_split_outside_event()
    test_always_on_nonfall_has_full_duty()
    test_duration_threshold_avoids_always_on_solution()
    test_candidate_grid_is_locked()
    test_esra_loss_is_finite_and_differentiable()
    test_one_epoch_candidate_training_smoke()
    print("FALL_ESRA_HEAD_V6_CORE_TESTS_OK")


if __name__ == "__main__":
    main()

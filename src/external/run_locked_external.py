from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon


SAMPLE_RATE_HZ = 50.0
WINDOW_LENGTH = 200
STRIDE_SAMPLES = 5
GRAVITY_M_S2 = 9.80665
PROXY_HALF_WIDTH_SECONDS = 1.5
GRACE_SECONDS = 0.5
DATASET_EXPECTATIONS = {
    "umafall": {
        "subjects": 19,
        "fall_trials": 208,
        "nonfall_trials": 538,
        "total_trials": 746,
    },
    "smartfallmm": {
        "subjects": 57,
        "fall_trials": 658,
        "nonfall_trials": 1656,
        "total_trials": 2314,
    },
}
METHOD_ORDER = (
    "fixed_0.5",
    "max_f1",
    "sensitivity_constrained",
    "srrc",
)
PRIMARY_METRICS = (
    "impact_proxy_event_recall",
    "mean_subject_false_alarms_per_hour",
    "pooled_false_alarms_per_hour",
    "worst_subject_false_alarms_per_hour",
    "cvar80_subject_false_alarms_per_hour",
    "proxy_median_delay_seconds",
    "proxy_p90_delay_seconds",
    "median_impact_relative_alarm_seconds",
    "p90_impact_relative_alarm_seconds",
)
EXPECTED_LOCKED_PARAMS = {
    "fixed_0.5": {"tau_on": 0.5, "tau_off": 0.5, "k": 1, "refractory": 2.0},
    "max_f1": {"tau_on": 0.85, "tau_off": 0.85, "k": 1, "refractory": 2.0},
    "sensitivity_constrained": {"tau_on": 0.9, "tau_off": 0.9, "k": 1, "refractory": 2.0},
    "srrc": {"tau_on": 0.85, "tau_off": 0.75, "k": 3, "refractory": 2.0},
}


@dataclass(frozen=True)
class Params:
    tau_on: float
    tau_off: float
    k: int
    refractory: float


@dataclass
class Trial:
    subject: str
    scene: str
    trial_id: str
    is_fall: bool
    sample_count: int
    duration_seconds: float
    impact_index: int | None
    impact_time: float | None
    proxy_start: float | None
    proxy_end: float | None
    signal: np.ndarray
    score_times: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked UMAFall/SmartFallMM validation for WEDA-trained UniMTS models."
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_EXPECTATIONS), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--unimts-code-dir", type=Path, required=True)
    parser.add_argument("--gate2a-code-dir", type=Path, required=True)
    parser.add_argument("--released-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--global-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--joint-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit inputs and configuration without model inference.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_clean(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def load_locked_params(path: Path) -> tuple[dict[str, Params], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = payload.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("global lock has no methods object")
    parsed: dict[str, Params] = {}
    for method in METHOD_ORDER:
        raw = methods.get(method, {}).get("params")
        if not isinstance(raw, dict):
            raise ValueError(f"global lock is missing {method}.params")
        normalized = {
            "tau_on": float(raw["tau_on"]),
            "tau_off": float(raw["tau_off"]),
            "k": int(raw["k"]),
            "refractory": float(raw["refractory"]),
        }
        if normalized != EXPECTED_LOCKED_PARAMS[method]:
            raise ValueError(
                f"locked parameters changed for {method}: {normalized} != "
                f"{EXPECTED_LOCKED_PARAMS[method]}"
            )
        parsed[method] = Params(**normalized)
    return parsed, payload


def discover_checkpoints(root: Path) -> list[dict[str, Any]]:
    pattern = re.compile(r"fold(?P<fold>[0-4])[/\\]acc_seed(?P<seed>[0-2])$")
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("fold*/acc_seed*/best_model.pth")):
        match = pattern.search(str(path.parent))
        if match:
            records.append(
                {
                    "fold": int(match.group("fold")),
                    "seed": int(match.group("seed")),
                    "model_id": f"fold{match.group('fold')}_acc_seed{match.group('seed')}",
                    "path": path,
                }
            )
    records.sort(key=lambda item: (item["fold"], item["seed"]))
    expected = [(fold, seed) for fold in range(5) for seed in range(3)]
    actual = [(item["fold"], item["seed"]) for item in records]
    if actual != expected:
        raise RuntimeError(f"expected 15 WEDA checkpoints {expected}, found {actual}")
    return records


UMAFALL_NAME = re.compile(
    r"^UMAFall_Subject_(?P<subject>\d+)_(?P<kind>ADL|Fall)_"
    r"(?P<scene>.+)_(?P<trial>\d+)_(?P<stamp>[0-9-]+_[0-9-]+)\.csv$"
)
SMARTFALL_NAME = re.compile(
    r"^S(?P<subject>\d+)A(?P<activity>\d+)T(?P<trial>\d+)\.csv$"
)
SMARTFALL_YOUNG_FALLS = set(range(10, 15))


def resample_xyz(times: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    if len(times) < 2 or xyz.shape != (len(times), 3):
        raise ValueError(f"invalid signal for resampling: times={times.shape}, xyz={xyz.shape}")
    order = np.argsort(times, kind="stable")
    times = times[order]
    xyz = xyz[order]
    unique_times, unique_indices = np.unique(times, return_index=True)
    xyz = xyz[unique_indices]
    if len(unique_times) < 2 or unique_times[-1] - unique_times[0] < 4.0:
        raise ValueError("trial has less than four seconds of valid acceleration")
    relative = unique_times - unique_times[0]
    target = np.arange(0.0, relative[-1] + 0.5 / SAMPLE_RATE_HZ, 1.0 / SAMPLE_RATE_HZ)
    result = np.column_stack(
        [np.interp(target, relative, xyz[:, axis]) for axis in range(3)]
    )
    return result.astype(np.float32)


def load_umafall_csv(path: Path) -> tuple[str, str, bool, np.ndarray, dict[str, Any]]:
    match = UMAFALL_NAME.match(path.name)
    if not match:
        raise ValueError(f"unexpected UMAFall filename: {path.name}")
    wrist_id: int | None = None
    times_ms: list[float] = []
    xyz_g: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("%") and ";" in line and "WRIST" in line.upper():
                fields = [value.strip() for value in line[1:].split(";")]
                if len(fields) >= 3 and fields[1].isdigit():
                    wrist_id = int(fields[1])
            elif line and not line.startswith("%") and wrist_id is not None:
                fields = line.split(";")
                if len(fields) < 7:
                    continue
                try:
                    timestamp = float(fields[0])
                    xyz = tuple(float(fields[index]) for index in (2, 3, 4))
                    sensor_type = int(float(fields[5]))
                    sensor_id = int(float(fields[6]))
                except ValueError:
                    continue
                if sensor_type == 0 and sensor_id == wrist_id:
                    times_ms.append(timestamp)
                    xyz_g.append(xyz)
    if wrist_id != 3:
        raise ValueError(f"{path}: expected corrected WRIST sensor ID 3, found {wrist_id}")
    times = np.asarray(times_ms, dtype=np.float64) / 1000.0
    raw_xyz = np.asarray(xyz_g, dtype=np.float64)
    if not np.isfinite(times).all() or not np.isfinite(raw_xyz).all():
        raise ValueError(f"{path}: non-finite wrist acceleration")
    signal = resample_xyz(times, raw_xyz * GRAVITY_M_S2)
    subject = f"subject_{int(match.group('subject')):02d}"
    scene = match.group("scene")
    is_fall = match.group("kind") == "Fall"
    metadata = {
        "raw_samples": len(raw_xyz),
        "raw_duration_seconds": float(times.max() - times.min()),
        "wrist_sensor_id": wrist_id,
    }
    return subject, scene, is_fall, signal, metadata


def load_smartfall_csv(path: Path, group: str) -> tuple[str, str, bool, np.ndarray, dict[str, Any]]:
    match = SMARTFALL_NAME.match(path.name)
    if not match:
        raise ValueError(f"unexpected SmartFallMM filename: {path.name}")
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
        for raw in handle:
            fields = [value.strip() for value in raw.strip().split(",")]
            if len(fields) not in (4, 5):
                continue
            try:
                values = [float(value) for value in fields[-3:]]
            except ValueError:
                continue
            if all(math.isfinite(value) for value in values):
                rows.append(values)
    if len(rows) < 128:
        raise ValueError(f"{path}: fewer than 128 valid samples")
    raw_xyz = np.asarray(rows, dtype=np.float64)
    times = np.arange(len(raw_xyz), dtype=np.float64) / 32.0
    signal = resample_xyz(times, raw_xyz)
    activity = int(match.group("activity"))
    if group == "old" and activity > 8:
        raise ValueError(f"{path}: old-group activity outside A01-A08")
    is_fall = group == "young" and activity in SMARTFALL_YOUNG_FALLS
    subject = f"{group}_S{int(match.group('subject')):02d}"
    scene = f"A{activity:02d}"
    metadata = {"raw_samples": len(raw_xyz), "nominal_sample_rate_hz": 32.0}
    return subject, scene, is_fall, signal, metadata


def load_external_trials(dataset: str, root: Path) -> tuple[list[Trial], dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    if dataset == "umafall":
        records = [(path, None) for path in sorted(root.glob("*.csv"))]
    else:
        records = []
        for group in ("young", "old"):
            directory = root / group / "accelerometer" / "watch"
            records.extend((path, group) for path in sorted(directory.glob("*.csv")))

    trials: list[Trial] = []
    exclusions: list[dict[str, str]] = []
    per_subject: dict[str, dict[str, int]] = {}
    per_scene: dict[str, int] = {}
    model_input_digest = hashlib.sha256()
    raw_samples = 0
    for path, group in records:
        try:
            if dataset == "umafall":
                subject, scene, is_fall, acc_m_s2, metadata = load_umafall_csv(path)
            else:
                assert group is not None
                subject, scene, is_fall, acc_m_s2, metadata = load_smartfall_csv(path, group)
        except ValueError as error:
            if dataset == "smartfallmm" and "fewer than 128" in str(error):
                exclusions.append({"file": str(path.relative_to(root)), "reason": "fewer_than_128_valid_samples"})
                continue
            raise
        raw_samples += int(metadata["raw_samples"])
        relative_path = str(path.relative_to(root)).replace("\\", "/")
        model_input_digest.update(relative_path.encode("utf-8"))
        model_input_digest.update(b"\0")
        model_input_digest.update(acc_m_s2.astype("<f4", copy=False).tobytes(order="C"))
        signal = np.zeros((len(acc_m_s2), 6), dtype=np.float32)
        signal[:, :3] = acc_m_s2
        duration = (len(signal) - 1) / SAMPLE_RATE_HZ
        if is_fall:
            impact_index = int(np.argmax(np.linalg.norm(acc_m_s2, axis=1)))
            impact_time = impact_index / SAMPLE_RATE_HZ
            proxy_start = max(0.0, impact_time - PROXY_HALF_WIDTH_SECONDS)
            proxy_end = min(duration, impact_time + PROXY_HALF_WIDTH_SECONDS)
        else:
            impact_index = None
            impact_time = None
            proxy_start = None
            proxy_end = None
        trials.append(
            Trial(
                subject=subject,
                scene=scene,
                trial_id=relative_path,
                is_fall=is_fall,
                sample_count=len(signal),
                duration_seconds=duration,
                impact_index=impact_index,
                impact_time=impact_time,
                proxy_start=proxy_start,
                proxy_end=proxy_end,
                signal=signal,
            )
        )
        stats = per_subject.setdefault(subject, {"fall": 0, "nonfall": 0})
        stats["fall" if is_fall else "nonfall"] += 1
        per_scene[scene] = per_scene.get(scene, 0) + 1

    observed = {
        "subjects": len(per_subject),
        "fall_trials": sum(item.is_fall for item in trials),
        "nonfall_trials": sum(not item.is_fall for item in trials),
        "total_trials": len(trials),
    }
    expected = DATASET_EXPECTATIONS[dataset]
    if observed != expected:
        raise RuntimeError(
            f"{dataset} audit mismatch: observed={observed}, expected={expected}; "
            f"excluded={len(exclusions)}"
        )
    trials.sort(key=lambda item: item.trial_id)
    impacts = [float(item.impact_time) for item in trials if item.impact_time is not None]
    audit = {
        "dataset": dataset,
        "root": str(root),
        **observed,
        "source_files": len(records),
        "excluded_files": exclusions,
        "raw_valid_samples": raw_samples,
        "model_sample_rate_hz": SAMPLE_RATE_HZ,
        "input_fields": ["wrist_acc_x", "wrist_acc_y", "wrist_acc_z"],
        "input_unit_conversion": (
            "g multiplied by 9.80665 to m/s^2"
            if dataset == "umafall"
            else "no conversion; source watch values treated as m/s^2"
        ),
        "model_input_sha256": model_input_digest.hexdigest(),
        "per_subject": per_subject,
        "per_scene": dict(sorted(per_scene.items())),
        "impact_time_seconds": {
            "min": float(np.min(impacts)),
            "median": float(np.median(impacts)),
            "max": float(np.max(impacts)),
        },
    }
    return trials, audit


def causal_windows(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if signal.ndim != 2 or signal.shape[1] != 6:
        raise ValueError(f"expected signal shape (T,6), got {signal.shape}")
    if len(signal) < WINDOW_LENGTH:
        pad = np.repeat(signal[:1], WINDOW_LENGTH - len(signal), axis=0)
        return np.concatenate([pad, signal], axis=0)[None], np.asarray([len(signal) - 1])
    ends = np.arange(WINDOW_LENGTH - 1, len(signal), STRIDE_SAMPLES, dtype=np.int64)
    if ends[-1] != len(signal) - 1:
        ends = np.concatenate([ends, np.asarray([len(signal) - 1], dtype=np.int64)])
    windows = np.stack([signal[end - WINDOW_LENGTH + 1 : end + 1] for end in ends])
    return windows.astype(np.float32), ends


def prepare_windows(trials: list[Trial]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    chunks: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for trial in trials:
        windows, ends = causal_windows(trial.signal)
        trial.score_times = ends.astype(np.float64) / SAMPLE_RATE_HZ
        chunks.append(windows)
        offsets.append((cursor, cursor + len(windows)))
        cursor += len(windows)
    return np.concatenate(chunks, axis=0), offsets


def batch_predict(
    model: torch.nn.Module,
    windows: np.ndarray,
    device: torch.device,
    joint_index: int,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
            logits = model(batch, joint_index)
            outputs.append(torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def alarm_segments(times: np.ndarray, scores: np.ndarray, params: Params) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    active = False
    consecutive = 0
    segment_start = math.nan
    next_allowed = -math.inf
    for time_value, score_value in zip(times, scores):
        t = float(time_value)
        score = float(score_value)
        if active:
            if score <= params.tau_off:
                result.append((segment_start, t))
                active = False
                next_allowed = t + params.refractory
            continue
        if t < next_allowed:
            consecutive = 0
            continue
        consecutive = consecutive + 1 if score >= params.tau_on else 0
        if consecutive >= params.k:
            segment_start = t
            active = True
            consecutive = 0
    if active and len(times):
        result.append((segment_start, float(times[-1])))
    return result


def evaluate_trial(trial: Trial, scores: np.ndarray, params: Params) -> dict[str, Any]:
    if trial.score_times is None:
        raise RuntimeError("score times were not prepared")
    segments = alarm_segments(trial.score_times, scores, params)
    detected_segments: list[tuple[float, float]] = []
    if trial.is_fall:
        assert trial.proxy_start is not None
        assert trial.proxy_end is not None
        detection_end = min(trial.duration_seconds, trial.proxy_end + GRACE_SECONDS)
        detected_segments = [
            segment
            for segment in segments
            if segment[1] >= trial.proxy_start and segment[0] <= detection_end
        ]
        false_segments = [segment for segment in segments if segment not in detected_segments]
        event_exposure = max(0.0, detection_end - trial.proxy_start)
        if detected_segments:
            first_start = min(segment[0] for segment in detected_segments)
            proxy_delay = max(first_start, trial.proxy_start) - trial.proxy_start
            impact_relative = first_start - float(trial.impact_time)
        else:
            first_start = None
            proxy_delay = None
            impact_relative = None
    else:
        detection_end = None
        false_segments = segments
        event_exposure = 0.0
        first_start = None
        proxy_delay = None
        impact_relative = None
    non_event_seconds = max(0.0, trial.duration_seconds - event_exposure)
    return {
        "subject": trial.subject,
        "scene": trial.scene,
        "trial_id": trial.trial_id,
        "is_fall": int(trial.is_fall),
        "sample_count": trial.sample_count,
        "duration_seconds": trial.duration_seconds,
        "impact_index": trial.impact_index,
        "impact_time_seconds": trial.impact_time,
        "proxy_start_seconds": trial.proxy_start,
        "proxy_end_seconds": trial.proxy_end,
        "grace_end_seconds": detection_end,
        "detected": int(bool(detected_segments)),
        "proxy_delay_seconds": proxy_delay,
        "first_detected_segment_start_seconds": first_start,
        "impact_relative_alarm_seconds": impact_relative,
        "false_alarms": len(false_segments),
        "non_event_hours": non_event_seconds / 3600.0,
        "alarm_count": len(segments),
        "alarm_segments_json": json.dumps(segments, separators=(",", ":")),
        "false_alarm_segments_json": json.dumps(false_segments, separators=(",", ":")),
    }


def empirical_cvar80(values: np.ndarray) -> float:
    if not len(values):
        return math.nan
    threshold = float(np.quantile(values, 0.8))
    return float(values[values >= threshold].mean())


def summarize_trials(details: list[dict[str, Any]], params: Params) -> dict[str, Any]:
    falls = [row for row in details if row["is_fall"] == 1]
    subject_totals: dict[str, dict[str, float]] = {}
    scenario_totals: dict[str, dict[str, float]] = {}
    for row in details:
        subject = subject_totals.setdefault(
            str(row["subject"]), {"false_alarms": 0.0, "non_event_hours": 0.0}
        )
        subject["false_alarms"] += float(row["false_alarms"])
        subject["non_event_hours"] += float(row["non_event_hours"])
        scene = scenario_totals.setdefault(
            str(row["scene"]),
            {"trials": 0.0, "fall_trials": 0.0, "detected_falls": 0.0, "false_alarms": 0.0, "non_event_hours": 0.0},
        )
        scene["trials"] += 1
        scene["fall_trials"] += int(row["is_fall"])
        scene["detected_falls"] += int(row["detected"])
        scene["false_alarms"] += float(row["false_alarms"])
        scene["non_event_hours"] += float(row["non_event_hours"])

    subject_rates = {
        subject: totals["false_alarms"] / totals["non_event_hours"]
        for subject, totals in subject_totals.items()
        if totals["non_event_hours"] > 0
    }
    rates = np.asarray(list(subject_rates.values()), dtype=np.float64)
    total_false = sum(float(row["false_alarms"]) for row in details)
    total_hours = sum(float(row["non_event_hours"]) for row in details)
    delays = np.asarray(
        [float(row["proxy_delay_seconds"]) for row in falls if row["proxy_delay_seconds"] is not None],
        dtype=np.float64,
    )
    impact_relative = np.asarray(
        [float(row["impact_relative_alarm_seconds"]) for row in falls if row["impact_relative_alarm_seconds"] is not None],
        dtype=np.float64,
    )
    per_scenario: dict[str, Any] = {}
    for scene, totals in sorted(scenario_totals.items()):
        hours = totals["non_event_hours"]
        fall_count = int(totals["fall_trials"])
        per_scenario[scene] = {
            "trials": int(totals["trials"]),
            "fall_trials": fall_count,
            "impact_proxy_detected_falls": int(totals["detected_falls"]),
            "impact_proxy_event_recall": totals["detected_falls"] / fall_count if fall_count else None,
            "false_alarms": int(totals["false_alarms"]),
            "non_event_hours": hours,
            "pooled_false_alarms_per_hour": totals["false_alarms"] / hours if hours > 0 else None,
        }
    return {
        "impact_proxy_event_recall": sum(int(row["detected"]) for row in falls) / len(falls),
        "mean_subject_false_alarms_per_hour": float(rates.mean()),
        "pooled_false_alarms_per_hour": total_false / total_hours,
        "worst_subject_false_alarms_per_hour": float(rates.max()),
        "cvar80_subject_false_alarms_per_hour": empirical_cvar80(rates),
        "proxy_median_delay_seconds": float(np.median(delays)) if len(delays) else math.nan,
        "proxy_p90_delay_seconds": float(np.percentile(delays, 90)) if len(delays) else math.nan,
        "median_impact_relative_alarm_seconds": float(np.median(impact_relative)) if len(impact_relative) else math.nan,
        "p90_impact_relative_alarm_seconds": float(np.percentile(impact_relative, 90)) if len(impact_relative) else math.nan,
        "fall_trials": len(falls),
        "detected_fall_trials": sum(int(row["detected"]) for row in falls),
        "all_trials": len(details),
        "subjects": len(subject_rates),
        "false_alarms": int(total_false),
        "non_event_hours": total_hours,
        "subject_false_alarms_per_hour": subject_rates,
        "per_scenario": per_scenario,
        "params": asdict(params),
    }


def write_window_predictions(
    path: Path,
    trials: list[Trial],
    offsets: list[tuple[int, int]],
    all_scores: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "subject", "scene", "trial_id", "is_fall", "score_time_seconds", "score",
        "impact_time_seconds", "impact_relative_time_seconds", "proxy_label",
    ]
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial, (start, end) in zip(trials, offsets):
            assert trial.score_times is not None
            for score_time, score in zip(trial.score_times, all_scores[start:end]):
                proxy_label = int(
                    trial.is_fall
                    and float(score_time) >= float(trial.proxy_start)
                    and float(score_time) <= float(trial.proxy_end)
                )
                writer.writerow(
                    {
                        "subject": trial.subject,
                        "scene": trial.scene,
                        "trial_id": trial.trial_id,
                        "is_fall": int(trial.is_fall),
                        "score_time_seconds": f"{float(score_time):.6f}",
                        "score": f"{float(score):.9g}",
                        "impact_time_seconds": "" if trial.impact_time is None else f"{trial.impact_time:.6f}",
                        "impact_relative_time_seconds": "" if trial.impact_time is None else f"{float(score_time) - trial.impact_time:.6f}",
                        "proxy_label": proxy_label,
                    }
                )
    temporary.replace(path)


def write_trial_details(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_clean(rows))
    temporary.replace(path)


def stats_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "n": len(finite),
        "mean": float(finite.mean()) if len(finite) else math.nan,
        "sample_std": float(finite.std(ddof=1)) if len(finite) > 1 else math.nan,
        "values": array.tolist(),
    }


def optional_float(value: Any) -> float:
    return math.nan if value is None else float(value)


def holm_adjust(p_values: list[float]) -> list[float]:
    result = [math.nan] * len(p_values)
    finite_indices = [index for index, value in enumerate(p_values) if math.isfinite(value)]
    ordered = sorted(finite_indices, key=lambda index: p_values[index])
    running = 0.0
    count = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * p_values[index])
        running = max(running, adjusted)
        result[index] = running
    return result


def paired_comparison(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    t_p_values: list[float] = []
    w_p_values: list[float] = []
    for metric in PRIMARY_METRICS:
        srrc = []
        baseline = []
        for result in model_results:
            srrc.append(optional_float(result["methods"]["srrc"][metric]))
            baseline.append(optional_float(result["methods"]["sensitivity_constrained"][metric]))
        srrc_array = np.asarray(srrc, dtype=np.float64)
        baseline_array = np.asarray(baseline, dtype=np.float64)
        valid = np.isfinite(srrc_array) & np.isfinite(baseline_array)
        srrc_array = srrc_array[valid]
        baseline_array = baseline_array[valid]
        differences = srrc_array - baseline_array
        if len(differences) > 1 and not np.allclose(differences, differences[0]):
            t_p = float(ttest_rel(srrc_array, baseline_array).pvalue)
        elif len(differences) > 1 and not np.allclose(differences, 0.0):
            t_p = 0.0
        else:
            t_p = 1.0
        if len(differences) and not np.allclose(differences, 0.0):
            try:
                w_p = float(wilcoxon(differences, alternative="two-sided", zero_method="wilcox").pvalue)
            except ValueError:
                w_p = math.nan
        else:
            w_p = 1.0
        comparisons[metric] = {
            "difference_definition": "SRRC minus sensitivity_constrained",
            "n_pairs": len(differences),
            "mean_difference": float(differences.mean()) if len(differences) else math.nan,
            "sample_std_difference": float(differences.std(ddof=1)) if len(differences) > 1 else math.nan,
            "positive": int(np.sum(differences > 1e-12)),
            "zero": int(np.sum(np.abs(differences) <= 1e-12)),
            "negative": int(np.sum(differences < -1e-12)),
            "differences": differences.tolist(),
            "paired_t_p": t_p,
            "wilcoxon_p": w_p,
        }
        t_p_values.append(t_p)
        w_p_values.append(w_p)
    for metric, t_adjusted, w_adjusted in zip(
        PRIMARY_METRICS, holm_adjust(t_p_values), holm_adjust(w_p_values)
    ):
        comparisons[metric]["paired_t_holm_p"] = t_adjusted
        comparisons[metric]["wilcoxon_holm_p"] = w_adjusted
    return comparisons


def aggregate_results(
    output_dir: Path,
    model_results: list[dict[str, Any]],
    protocol: dict[str, Any],
    data_audit: dict[str, Any],
) -> dict[str, Any]:
    dataset = str(data_audit["dataset"])
    by_method: dict[str, Any] = {}
    for method in METHOD_ORDER:
        by_method[method] = {
            metric: stats_summary(result["methods"][method][metric] for result in model_results)
            for metric in PRIMARY_METRICS
        }
    scenes = sorted(
        {
            scene
            for result in model_results
            for method in METHOD_ORDER
            for scene in result["methods"][method]["per_scenario"]
        }
    )
    per_scenario_by_method: dict[str, Any] = {}
    for method in METHOD_ORDER:
        per_scenario_by_method[method] = {}
        for scene in scenes:
            rows = [result["methods"][method]["per_scenario"][scene] for result in model_results]
            per_scenario_by_method[method][scene] = {
                metric: stats_summary(optional_float(row[metric]) for row in rows)
                for metric in (
                    "impact_proxy_event_recall",
                    "false_alarms",
                    "pooled_false_alarms_per_hour",
                )
            }
    aggregate = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "external_dataset_used_for_tuning": False,
        "models": len(model_results),
        "protocol": protocol,
        "data_audit": data_audit,
        "by_method": by_method,
        "per_scenario_by_method": per_scenario_by_method,
        "paired_srrc_vs_sensitivity_constrained": paired_comparison(model_results),
        "model_results": model_results,
    }
    save_json(output_dir / f"{dataset}_external_aggregate.json", aggregate)

    row_fields = ["model_id", "fold", "seed", "method", *PRIMARY_METRICS]
    with (output_dir / f"{dataset}_external_model_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        for result in model_results:
            for method in METHOD_ORDER:
                writer.writerow(
                    {
                        "model_id": result["model_id"],
                        "fold": result["fold"],
                        "seed": result["seed"],
                        "method": method,
                        **{metric: result["methods"][method][metric] for metric in PRIMARY_METRICS},
                    }
                )

    with (output_dir / f"{dataset}_external_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "metric", "n", "mean", "sample_std"])
        writer.writeheader()
        for method in METHOD_ORDER:
            for metric in PRIMARY_METRICS:
                stats = by_method[method][metric]
                writer.writerow(
                    {
                        "method": method,
                        "metric": metric,
                        "n": stats["n"],
                        "mean": stats["mean"],
                        "sample_std": stats["sample_std"],
                    }
                )

    with (output_dir / f"{dataset}_external_per_scenario.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["method", "scene", "metric", "n", "mean", "sample_std"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHOD_ORDER:
            for scene in scenes:
                for metric, stats in per_scenario_by_method[method][scene].items():
                    writer.writerow(
                        {
                            "method": method,
                            "scene": scene,
                            "metric": metric,
                            "n": stats["n"],
                            "mean": stats["mean"],
                            "sample_std": stats["sample_std"],
                        }
                    )

    markdown = [
        f"# Locked {dataset} External Validation",
        "",
        f"{dataset} was not used to select a model, threshold, event window, or controller parameter.",
        "Event results use an accelerometer-impact proxy and are not manual event annotations.",
        "",
        "| Method | Impact-proxy recall | Mean subject FA/h | Worst subject FA/h | CVaR80 FA/h | Proxy median delay (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        method_stats = by_method[method]
        markdown.append(
            "| {method} | {recall:.4f} +/- {recall_std:.4f} | {mean_fa:.2f} +/- {mean_fa_std:.2f} | "
            "{worst:.2f} +/- {worst_std:.2f} | {cvar:.2f} +/- {cvar_std:.2f} | {delay:.3f} +/- {delay_std:.3f} |".format(
                method=method,
                recall=method_stats["impact_proxy_event_recall"]["mean"],
                recall_std=method_stats["impact_proxy_event_recall"]["sample_std"],
                mean_fa=method_stats["mean_subject_false_alarms_per_hour"]["mean"],
                mean_fa_std=method_stats["mean_subject_false_alarms_per_hour"]["sample_std"],
                worst=method_stats["worst_subject_false_alarms_per_hour"]["mean"],
                worst_std=method_stats["worst_subject_false_alarms_per_hour"]["sample_std"],
                cvar=method_stats["cvar80_subject_false_alarms_per_hour"]["mean"],
                cvar_std=method_stats["cvar80_subject_false_alarms_per_hour"]["sample_std"],
                delay=method_stats["proxy_median_delay_seconds"]["mean"],
                delay_std=method_stats["proxy_median_delay_seconds"]["sample_std"],
            )
        )
    markdown.extend(
        [
            "",
            "Paired differences are defined as SRRC minus sensitivity-constrained over all 15 locked WEDA models.",
            "Both raw two-sided p-values and Holm-adjusted p-values are stored in the aggregate JSON.",
            "",
        ]
    )
    (output_dir / f"{dataset}_external_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    return aggregate


def environment_payload(device: torch.device) -> dict[str, Any]:
    payload = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "device_requested": str(device),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        payload["device_name"] = torch.cuda.get_device_name(device)
        payload["device_count"] = torch.cuda.device_count()
    return payload


def protocol_payload(dataset: str, params: dict[str, Params]) -> dict[str, Any]:
    if dataset == "umafall":
        source = {
            "release": "UMAFall corrected version only",
            "eligibility": "all 746 corrected CSV trials",
            "source_time_axis": "WRIST SensorTag timestamps in milliseconds",
            "source_unit": "g",
            "conversion": "multiply by 9.80665 to m/s^2",
            "resampling": "linear interpolation on recorded timestamps to 50 Hz",
        }
    else:
        source = {
            "release": "SmartFallMM smartwatch accelerometer",
            "eligibility": "all young and old watch trials with at least 128 valid samples",
            "source_time_axis": "sample index at the documented nominal 32 Hz",
            "source_unit": "treated as m/s^2; no numeric conversion",
            "conversion": "none",
            "resampling": "linear interpolation from nominal 32 Hz to 50 Hz",
            "excluded": "files with fewer than 128 valid samples, fixed before inference",
        }
    return {
        "name": f"locked_{dataset}_external_validation_v1",
        "dataset": dataset,
        "external_dataset_used_for_tuning": False,
        "model_selection_on_external_data": False,
        "threshold_selection_on_external_data": False,
        "axis_search_on_external_data": False,
        "input_channels": ["wrist_acc_x", "wrist_acc_y", "wrist_acc_z"],
        "source_processing": source,
        "unused_model_channels": "channels 4-6 are zero padded; the Acc model reads channels 1-3",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_length_samples": WINDOW_LENGTH,
        "window_seconds": WINDOW_LENGTH / SAMPLE_RATE_HZ,
        "causal_window": True,
        "stride_samples": STRIDE_SAMPLES,
        "stride_seconds": STRIDE_SAMPLES / SAMPLE_RATE_HZ,
        "impact_proxy": "argmax L2 norm of [ax, ay, az] within known fall trials",
        "proxy_half_width_seconds": PROXY_HALF_WIDTH_SECONDS,
        "proxy_window_seconds": 2 * PROXY_HALF_WIDTH_SECONDS,
        "grace_seconds": GRACE_SECONDS,
        "duration_denominator": "full trial duration minus clipped proxy window plus grace for fall trials",
        "nonfall_policy": "all eligible trials whose published activity label is not a fall are retained",
        "models": "all 5 folds x 3 seeds WEDA Acc checkpoints",
        "locked_methods": {method: asdict(value) for method, value in params.items()},
        "statistical_unit": "15 locked WEDA model checkpoints",
        "paired_difference": "SRRC minus sensitivity_constrained",
        "multiplicity": "Holm adjustment across reported paired metrics, separately per test family",
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.released_checkpoint.is_file():
        raise FileNotFoundError(args.released_checkpoint)
    if not (args.gate2a_code_dir / "weda_common.py").is_file():
        raise FileNotFoundError(args.gate2a_code_dir / "weda_common.py")
    params, lock_payload = load_locked_params(args.global_lock)
    checkpoints = discover_checkpoints(args.checkpoint_root)
    trials, data_audit = load_external_trials(args.dataset, args.dataset_root)
    protocol = protocol_payload(args.dataset, params)
    protocol_hash = fingerprint(protocol)
    save_json(args.output_dir / "locked_protocol.json", protocol)
    save_json(args.output_dir / "data_audit.json", data_audit)
    save_json(args.output_dir / "global_lock_copy.json", lock_payload)
    save_json(args.output_dir / "environment.json", environment_payload(torch.device(args.device)))
    print(json.dumps({"data_audit": data_audit, "protocol_sha256": protocol_hash}, indent=2), flush=True)
    if args.dry_run:
        print(f"{args.dataset.upper()}_EXTERNAL_DRY_RUN_OK", flush=True)
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA/ROCm device, but torch.cuda.is_available() is False")

    windows, offsets = prepare_windows(trials)
    print(f"PREPARED_WINDOWS={len(windows)} shape={windows.shape}", flush=True)
    gate2a = str(args.gate2a_code_dir)
    if gate2a not in sys.path:
        sys.path.insert(0, gate2a)
    from weda_common import build_model_from_checkpoint

    device = torch.device(args.device)
    released_sha256 = sha256_file(args.released_checkpoint)
    model_results: list[dict[str, Any]] = []
    for index, record in enumerate(checkpoints, start=1):
        model_id = str(record["model_id"])
        checkpoint_path = Path(record["path"])
        model_dir = args.output_dir / "models" / model_id
        result_path = model_dir / "result.json"
        predictions_path = model_dir / "window_predictions.csv.gz"
        details_path = model_dir / "trial_details.csv.gz"
        complete_path = model_dir / "COMPLETE.json"
        required = (result_path, predictions_path, details_path, complete_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        if all(path.is_file() and path.stat().st_size > 0 for path in required):
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if complete.get("protocol_sha256") != protocol_hash:
                raise RuntimeError(f"{model_id}: completed output uses a different protocol")
            if complete.get("checkpoint_sha256") != checkpoint_sha256:
                raise RuntimeError(f"{model_id}: checkpoint changed since the completed output")
            if complete.get("released_checkpoint_sha256") != released_sha256:
                raise RuntimeError(f"{model_id}: released UniMTS checkpoint changed")
            if complete.get("external_model_input_sha256") != data_audit["model_input_sha256"]:
                raise RuntimeError(f"{model_id}: external model input changed")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            model_results.append(result)
            print(f"SKIP_COMPLETE {index}/15 model={model_id}", flush=True)
            continue

        print(f"START_MODEL {index}/15 model={model_id}", flush=True)
        model_started = time.time()
        model = build_model_from_checkpoint(
            variant="acc",
            checkpoint_path=checkpoint_path,
            unimts_code_dir=args.unimts_code_dir,
            released_checkpoint=args.released_checkpoint,
            device=device,
        )
        scores = batch_predict(model, windows, device, args.joint_index, args.batch_size)
        write_window_predictions(predictions_path, trials, offsets, scores)
        method_summaries: dict[str, Any] = {}
        detail_rows: list[dict[str, Any]] = []
        for method in METHOD_ORDER:
            method_details = []
            for trial, (start, end) in zip(trials, offsets):
                detail = evaluate_trial(trial, scores[start:end], params[method])
                method_details.append(detail)
                detail_rows.append({"method": method, **detail})
            method_summaries[method] = summarize_trials(method_details, params[method])
        write_trial_details(details_path, detail_rows)
        result = {
            "model_id": model_id,
            "fold": int(record["fold"]),
            "seed": int(record["seed"]),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "released_checkpoint": str(args.released_checkpoint),
            "released_checkpoint_sha256": released_sha256,
            "protocol_sha256": protocol_hash,
            "external_dataset_used_for_tuning": False,
            "methods": method_summaries,
            "elapsed_seconds": time.time() - model_started,
        }
        save_json(result_path, result)
        save_json(
            complete_path,
            {
                "status": "complete",
                "completed_at_utc": utc_now(),
                "model_id": model_id,
                "protocol_sha256": protocol_hash,
                "checkpoint_sha256": checkpoint_sha256,
                "released_checkpoint_sha256": released_sha256,
                "external_model_input_sha256": data_audit["model_input_sha256"],
            },
        )
        model_results.append(result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        save_json(
            args.output_dir / "progress.json",
            {
                "state": "running",
                "completed_models": len(model_results),
                "total_models": 15,
                "last_model": model_id,
                "updated_at_utc": utc_now(),
            },
        )
        print(
            f"COMPLETE_MODEL {index}/15 model={model_id} elapsed={time.time() - model_started:.1f}s",
            flush=True,
        )

    model_results.sort(key=lambda item: (int(item["fold"]), int(item["seed"])))
    aggregate_results(args.output_dir, model_results, protocol, data_audit)
    save_json(
        args.output_dir / "progress.json",
        {
            "state": "complete",
            "completed_models": 15,
            "total_models": 15,
            "elapsed_seconds": time.time() - started,
            "updated_at_utc": utc_now(),
        },
    )
    save_json(
        args.output_dir / f"{args.dataset.upper()}_EXTERNAL_COMPLETE.json",
        {
            "status": "complete",
            "models": 15,
            "protocol_sha256": protocol_hash,
            "external_dataset_used_for_tuning": False,
            "completed_at_utc": utc_now(),
        },
    )
    print(f"{args.dataset.upper()}_EXTERNAL_RESULTS={args.output_dir}", flush=True)
    print(f"{args.dataset.upper()}_EXTERNAL_VALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

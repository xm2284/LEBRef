from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    PRIMARY_SEED, add_runtime_args, atomic_json, load_head, load_linear_core, load_runtime,
    validate_runtime_paths, write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-retraining alarm-segment precision-recall and recall-burden curves.")
    add_runtime_args(p)
    p.add_argument("--datasets", nargs="+", default=["smartfallmm", "umafall"], choices=["smartfallmm", "umafall"])
    p.add_argument("--methods", nargs="+", default=["probe", "event_mil", "lebref"], choices=["probe", "event_mil", "lebref"])
    p.add_argument("--seed", type=int, default=PRIMARY_SEED)
    p.add_argument("--threshold-step", type=float, default=0.01)
    return p.parse_args()


def method_scores(linear_core: Any, head: Any, features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return linear_core.predict(head, features, indices)


def alarm_precision_from_contrib(contrib: dict[str, dict[str, Any]]) -> tuple[float, int, int]:
    tp = int(sum(int(r["detected_observable"]) for r in contrib.values()))
    fp = int(round(sum(float(r["false_alarms"]) for r in contrib.values())))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    return float(precision), tp, fp


def compute_curve(args: argparse.Namespace, dataset: str, method: str, thresholds: np.ndarray) -> list[dict[str, Any]]:
    v6_runner, v6_core, runtime = load_runtime(args, dataset)
    linear_core = load_linear_core(Path(args.linear_core))
    details_by_threshold: dict[float, list[dict[str, Any]]] = {float(t): [] for t in thresholds}
    for fold in range(5):
        _, _, test_subjects = v6_runner.split_subjects(runtime.folds, fold)
        test_trials, test_offsets, test_indices = runtime.legacy.subset(runtime.trials, runtime.offsets, test_subjects)
        head = load_head(linear_core, Path(args.v9_results_dir), dataset, fold, method, args.seed, args.device)
        scores = method_scores(linear_core, head, runtime.features, test_indices)
        for threshold in thresholds:
            _, details = v6_core.evaluate_duration(runtime.external, test_trials, test_offsets, scores, float(threshold))
            details_by_threshold[float(threshold)].extend(details)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        details = details_by_threshold[float(threshold)]
        contrib = v6_core.subject_contributions(details)
        summary = v6_core.aggregate_subject_sample(contrib, sorted(contrib))
        precision, tp, fp = alarm_precision_from_contrib(contrib)
        rows.append({
            "dataset": dataset, "method": method, "seed": args.seed, "threshold": float(threshold),
            "observable_event_recall": float(summary["observable_event_recall"]),
            "alarm_segment_precision": precision,
            "true_positive_observable_events": tp,
            "false_alarm_segments": fp,
            "mean_subject_false_alarm_duty": float(summary["mean_subject_false_alarm_duty_cycle"]),
            "pooled_false_alarm_duty": float(summary["pooled_false_alarm_duty_cycle"]),
            "mean_subject_fa_per_hour": float(summary["mean_subject_false_alarms_per_hour"]),
            "pooled_fa_per_hour": float(summary["pooled_false_alarms_per_hour"]),
        })
    return rows


def plot_curves(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    methods = ["probe", "event_mil", "lebref"]
    labels = {"probe":"Linear probe", "event_mil":"Event-MIL", "lebref":"LEBRef"}
    for dataset in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        for method in methods:
            m = sorted([r for r in sub if r["method"] == method], key=lambda r: r["observable_event_recall"])
            if m:
                ax.plot([r["observable_event_recall"] for r in m], [r["alarm_segment_precision"] for r in m], marker="o", markersize=2.5, linewidth=1.1, label=labels[method])
        ax.set_xlabel("Observable-event recall")
        ax.set_ylabel("Alarm-segment precision")
        ax.set_title(f"{dataset}: alarm-segment precision–recall")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"pr_{dataset}.pdf")
        fig.savefig(out_dir / f"pr_{dataset}.png", dpi=240)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        for method in methods:
            m = sorted([r for r in sub if r["method"] == method], key=lambda r: r["observable_event_recall"])
            if m:
                ax.plot([r["observable_event_recall"] for r in m], [100*r["mean_subject_false_alarm_duty"] for r in m], marker="o", markersize=2.5, linewidth=1.1, label=labels[method])
        ax.set_xlabel("Observable-event recall")
        ax.set_ylabel("Mean false-alarm duty (%)")
        ax.set_title(f"{dataset}: recall–duty operating curve")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"recall_duty_{dataset}.pdf")
        fig.savefig(out_dir / f"recall_duty_{dataset}.png", dpi=240)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_runtime_paths(args)
    out_dir = Path(args.output_dir) / "final_new_analyses" / "pr_operating_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = np.arange(args.threshold_step, 1.0, args.threshold_step, dtype=float)
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for method in args.methods:
            print(f"CURVE dataset={dataset} method={method} seed={args.seed}", flush=True)
            rows.extend(compute_curve(args, dataset, method, thresholds))
    write_csv(out_dir / "alarm_segment_pr_and_burden_curves.csv", rows)
    atomic_json(out_dir / "curve_manifest.json", {
        "state": "complete", "analysis": "no-retraining common-threshold curves",
        "precision_definition": "detected observable fall events / (detected observable fall events + fully non-overlapping false alarm segments)",
        "recall_definition": "detected observable fall events / observable fall events",
        "important": "This is alarm-segment precision, not window-level classification precision.",
        "seed": args.seed, "thresholds": [float(x) for x in thresholds],
    })
    plot_curves(rows, out_dir)
    print("PR_OPERATING_CURVE_ANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

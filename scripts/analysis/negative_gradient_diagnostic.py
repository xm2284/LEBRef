from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    PRIMARY_SEED, SEEDS, add_runtime_args, atomic_json, bootstrap_mean, load_head,
    load_linear_core, load_runtime, method_threshold, validate_runtime_paths, write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-retraining negative-bag segmented gradient attribution.")
    add_runtime_args(p)
    p.add_argument("--datasets", nargs="+", default=["smartfallmm", "umafall"], choices=["smartfallmm", "umafall"])
    p.add_argument("--methods", nargs="+", default=["event_mil", "lebref"], choices=["event_mil", "lebref"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--hard-top-fraction", type=float, default=0.01)
    p.add_argument("--sustained-threshold-fraction", type=float, default=0.50)
    p.add_argument("--sustained-min-run", type=int, default=5, help="windows; 5 = 0.5 s at 0.1-s score step")
    p.add_argument("--sensitivity", action="store_true", help="Run a small definition-sensitivity grid in addition to the primary diagnostic.")
    return p.parse_args()


def contiguous_run_mask(mask: np.ndarray, min_run: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    start = None
    values = mask.tolist() + [False]
    for i, v in enumerate(values):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_run:
                out[start:i] = True
            start = None
    return out


def classify_regions(probs: np.ndarray, threshold: float, hard_frac: float, sustained_frac: float, min_run: int) -> np.ndarray:
    n = len(probs)
    if n == 0:
        return np.asarray([], dtype=np.int8)
    k = max(1, int(math.ceil(n * hard_frac)))
    top = np.argpartition(probs, max(0, n-k))[n-k:]
    hard = np.zeros(n, dtype=bool)
    hard[top] = True
    moderate = probs >= float(sustained_frac * threshold)
    sustained = contiguous_run_mask(moderate, min_run) & ~hard
    labels = np.zeros(n, dtype=np.int8)  # 0 remaining, 1 sustained, 2 hard
    labels[sustained] = 1
    labels[hard] = 2
    return labels


def bag_gradients_from_logits(logits_np: np.ndarray, temperature: float) -> tuple[np.ndarray, float]:
    logits = torch.as_tensor(logits_np, dtype=torch.float64).detach().requires_grad_(True)
    probs = torch.sigmoid(logits)
    weights = torch.softmax(probs / temperature, dim=0)
    score = torch.sum(weights * probs)
    loss = -torch.log((1.0 - score).clamp(1e-12, 1.0))
    loss.backward()
    grad = logits.grad.detach().cpu().numpy().astype(np.float64)
    return grad, float(loss.detach().cpu())


def head_logits(head: Any, features: np.ndarray, indices: np.ndarray, device: str) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(features[indices], dtype=np.float32), device=device)
        normalized = (x - head.feature_mean) / head.feature_scale
        logits = head.linear(normalized).squeeze(-1)
    return logits.detach().cpu().numpy().astype(np.float64)


def collect_trial_cache(args: argparse.Namespace, dataset: str, method: str, seed: int, v6_runner: Any, runtime: Any, linear_core: Any) -> list[dict[str, Any]]:
    cache: list[dict[str, Any]] = []
    for fold in range(5):
        _, _, test_subjects = v6_runner.split_subjects(runtime.folds, fold)
        test_trials, test_offsets, test_indices = runtime.legacy.subset(runtime.trials, runtime.offsets, test_subjects)
        head = load_head(linear_core, Path(args.v9_results_dir), dataset, fold, method, seed, args.device)
        threshold = method_threshold(Path(args.v9_results_dir), dataset, fold, method, seed)
        for trial, (local_start, local_end) in zip(test_trials, test_offsets):
            if bool(trial.is_fall):
                continue
            indices = test_indices[local_start:local_end]
            if len(indices) == 0:
                continue
            logits = head_logits(head, runtime.features, indices, args.device)
            probs = 1.0 / (1.0 + np.exp(-logits))
            grad, loss = bag_gradients_from_logits(logits, args.temperature)
            cache.append({
                "dataset": dataset, "method": method, "seed": seed, "fold": fold,
                "subject": str(trial.subject), "trial_id": str(getattr(trial, "trial_id", getattr(trial, "source_id", ""))),
                "threshold": threshold, "probs": probs, "grad": grad, "loss": loss,
            })
    return cache


def rows_for_setting(cache: list[dict[str, Any]], args: argparse.Namespace, hard_frac: float, sustained_frac: float, min_run: int, setting_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    region_names = {0: "remaining_background", 1: "sustained_activation", 2: "hard_trigger"}
    for item in cache:
        probs, grad = item["probs"], item["grad"]
        labels = classify_regions(probs, item["threshold"], hard_frac, sustained_frac, min_run)
        abs_total = float(np.abs(grad).sum())
        signed_total = float(grad.sum())
        for code, name in region_names.items():
            m = labels == code
            count = int(m.sum())
            out.append({
                "dataset": item["dataset"], "method": item["method"], "seed": item["seed"], "fold": item["fold"],
                "subject": item["subject"], "trial_id": item["trial_id"], "setting": setting_name,
                "selected_threshold": item["threshold"], "temperature": args.temperature,
                "hard_top_fraction": hard_frac, "sustained_threshold_fraction": sustained_frac, "sustained_min_run_windows": min_run,
                "region": name, "window_count": count, "window_share": count / len(labels),
                "abs_gradient_sum": float(np.abs(grad[m]).sum()),
                "abs_gradient_share": float(np.abs(grad[m]).sum() / abs_total) if abs_total > 0 else float("nan"),
                "signed_gradient_sum": float(grad[m].sum()),
                "signed_gradient_share": float(grad[m].sum() / signed_total) if signed_total != 0 else float("nan"),
                "negative_bag_loss": item["loss"], "max_probability": float(probs.max()), "mean_probability": float(probs.mean()),
                "n_windows": len(labels),
            })
    return out


def aggregate_subject_equal(rows: list[dict[str, Any]], iterations: int) -> list[dict[str, Any]]:
    keys = sorted({(r["dataset"], r["method"], r["seed"], r["setting"], r["region"]) for r in rows})
    output: list[dict[str, Any]] = []
    for key in keys:
        subset = [r for r in rows if (r["dataset"], r["method"], r["seed"], r["setting"], r["region"]) == key]
        by_subject: dict[str, list[dict[str, Any]]] = {}
        for r in subset:
            by_subject.setdefault(r["subject"], []).append(r)
        subject_abs = [float(np.nanmean([x["abs_gradient_share"] for x in rs])) for rs in by_subject.values()]
        subject_win = [float(np.nanmean([x["window_share"] for x in rs])) for rs in by_subject.values()]
        a = bootstrap_mean(subject_abs, iterations, 20260812 + int(key[2]) % 1000)
        w = bootstrap_mean(subject_win, iterations, 20260813 + int(key[2]) % 1000)
        output.append({
            "dataset": key[0], "method": key[1], "seed": key[2], "setting": key[3], "region": key[4],
            "n_subjects": a["n"], "mean_abs_gradient_share": a["mean"], "grad_ci_low": a["ci_low"], "grad_ci_high": a["ci_high"],
            "mean_window_share": w["mean"], "window_ci_low": w["ci_low"], "window_ci_high": w["ci_high"],
        })
    return output


def plot_primary(summary: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    primary = [r for r in summary if r["seed"] == PRIMARY_SEED and r["setting"] == "primary"]
    regions = ["hard_trigger", "sustained_activation", "remaining_background"]
    for dataset in sorted({r["dataset"] for r in primary}):
        for method in sorted({r["method"] for r in primary if r["dataset"] == dataset}):
            rows = {r["region"]: r for r in primary if r["dataset"] == dataset and r["method"] == method}
            if not all(x in rows for x in regions):
                continue
            y = np.asarray([rows[x]["mean_abs_gradient_share"] for x in regions])
            lo = np.asarray([rows[x]["grad_ci_low"] for x in regions])
            hi = np.asarray([rows[x]["grad_ci_high"] for x in regions])
            err = np.vstack([y-lo, hi-y])
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            ax.bar(np.arange(3), y, yerr=err, capsize=4)
            ax.set_xticks(np.arange(3), ["Hard trigger", "Sustained", "Remaining"])
            ax.set_ylabel("Subject-equal gradient share")
            ax.set_ylim(0, max(1.0, float(hi.max())*1.12))
            ax.set_title(f"{dataset}: {method} negative-bag gradient attribution")
            fig.tight_layout()
            fig.savefig(out_dir / f"gradient_share_{dataset}_{method}.pdf")
            fig.savefig(out_dir / f"gradient_share_{dataset}_{method}.png", dpi=240)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_runtime_paths(args)
    out_dir = Path(args.output_dir) / "final_new_analyses" / "negative_gradient"
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = [(args.hard_top_fraction, args.sustained_threshold_fraction, args.sustained_min_run, "primary")]
    if args.sensitivity:
        for hf, sf, mr in [(0.05,0.50,5),(0.01,0.40,5),(0.01,0.60,5),(0.01,0.50,3),(0.01,0.50,10)]:
            settings.append((hf, sf, mr, f"hf{hf:.2f}_sf{sf:.2f}_mr{mr}"))
    all_rows: list[dict[str, Any]] = []
    linear_core = load_linear_core(Path(args.linear_core))
    for dataset in args.datasets:
        v6_runner, _, runtime = load_runtime(args, dataset)
        for method in args.methods:
            for seed in args.seeds:
                print(f"GRADIENT_CACHE dataset={dataset} method={method} seed={seed}", flush=True)
                cache = collect_trial_cache(args, dataset, method, seed, v6_runner, runtime, linear_core)
                for hf, sf, mr, name in settings:
                    all_rows.extend(rows_for_setting(cache, args, hf, sf, mr, name))
    summary = aggregate_subject_equal(all_rows, args.bootstrap_iterations)
    write_csv(out_dir / "negative_gradient_trial_regions.csv", all_rows)
    write_csv(out_dir / "negative_gradient_subject_equal_summary.csv", summary)
    atomic_json(out_dir / "negative_gradient_manifest.json", {
        "state": "complete", "analysis": "direct_negative_bag_gradient_attribution_no_retraining",
        "primary_definition": {"hard_top_fraction": args.hard_top_fraction, "sustained_threshold_fraction": args.sustained_threshold_fraction, "sustained_min_run_windows": args.sustained_min_run},
        "interpretation": "Gradient shares are diagnostic, not causal effects. Region definitions are prespecified and optionally sensitivity-analyzed.",
        "rows": len(all_rows), "summary_rows": len(summary), "seeds": args.seeds,
    })
    plot_primary(summary, out_dir)
    print("NEGATIVE_GRADIENT_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

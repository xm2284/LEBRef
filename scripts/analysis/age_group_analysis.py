from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    SEEDS, add_runtime_args, atomic_json, bootstrap_mean, diff_in_group_means_bootstrap,
    load_oof_details, subject_false_alarm_duty, validate_runtime_paths, write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-retraining SmartFallMM age-group false-alarm analysis.")
    add_runtime_args(p)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return p.parse_args()


def age_group(subject: str) -> str:
    if subject.startswith("old_"):
        return "older"
    if subject.startswith("young_"):
        return "younger"
    return "unknown"


def main() -> None:
    args = parse_args()
    validate_runtime_paths(args, need_v9=True)
    out_dir = Path(args.output_dir) / "final_new_analyses" / "age_group"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = []
    did = []
    for seed in args.seeds:
        probe = subject_false_alarm_duty(load_oof_details(Path(args.v9_results_dir), "smartfallmm", "probe", seed))
        leb = subject_false_alarm_duty(load_oof_details(Path(args.v9_results_dir), "smartfallmm", "lebref", seed))
        common = sorted(set(probe) & set(leb))
        groups = {s: age_group(s) for s in common}
        diff = {s: leb[s] - probe[s] for s in common}
        for s in common:
            rows.append({
                "seed": seed, "subject": s, "age_group": groups[s],
                "probe_duty": probe[s], "lebref_duty": leb[s], "delta_duty": diff[s],
            })
        for group in ("younger", "older"):
            members = [s for s in common if groups[s] == group]
            pstats = bootstrap_mean([probe[s] for s in members], args.bootstrap_iterations, 20260812 + seed % 1000 + (0 if group == "younger" else 10))
            lstats = bootstrap_mean([leb[s] for s in members], args.bootstrap_iterations, 20260813 + seed % 1000 + (0 if group == "younger" else 10))
            dstats = bootstrap_mean([diff[s] for s in members], args.bootstrap_iterations, 20260814 + seed % 1000 + (0 if group == "younger" else 10))
            summary.append({
                "seed": seed, "age_group": group, "n_subjects": len(members),
                "probe_mean_duty": pstats["mean"], "lebref_mean_duty": lstats["mean"],
                "mean_delta_duty": dstats["mean"], "delta_ci_low": dstats["ci_low"], "delta_ci_high": dstats["ci_high"],
            })
        did.append({"seed": seed, **diff_in_group_means_bootstrap(diff, groups, "older", "younger", args.bootstrap_iterations, 20260815 + seed % 1000)})
    write_csv(out_dir / "smartfallmm_age_subjects.csv", rows)
    write_csv(out_dir / "smartfallmm_age_group_summary.csv", summary)
    write_csv(out_dir / "smartfallmm_age_difference_in_improvement.csv", did)
    atomic_json(out_dir / "age_group_manifest.json", {
        "state": "complete",
        "analysis": "SmartFallMM age-group false-alarm burden, no retraining",
        "scope": "false-alarm duty only; fall recall is not compared across age groups because fall trials are contributed by the younger cohort",
        "caution": "Observational subgroup analysis; do not interpret as a causal age effect.",
        "expected_prefixes": ["young_", "old_"],
    })
    print("AGE_GROUP_ANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

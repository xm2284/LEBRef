from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import atomic_json


def read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt(x, digits=4):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--known-results", required=True)
    args = p.parse_args()
    root = Path(args.output_dir)
    known = json.loads(Path(args.known_results).read_text(encoding="utf-8"))
    grad = read_csv(root / "final_new_analyses/negative_gradient/negative_gradient_subject_equal_summary.csv")
    age = read_csv(root / "final_new_analyses/age_group/smartfallmm_age_group_summary.csv")
    age_did = read_csv(root / "final_new_analyses/age_group/smartfallmm_age_difference_in_improvement.csv")
    intensity = read_csv(root / "final_new_analyses/motion_intensity/motion_intensity_tercile_summary.csv")
    intensity_corr = read_csv(root / "final_new_analyses/motion_intensity/motion_intensity_correlation.csv")
    curve = read_csv(root / "final_new_analyses/pr_operating_curves/alarm_segment_pr_and_burden_curves.csv")

    lines = []
    lines.append("# LEBRef Final Experimental Closure Report (V19)\n")
    lines.append("**Run type:** no-retraining final closure. Existing training experiments are treated as locked; this run adds only the remaining diagnostics requested by the teachers.\n")
    lines.append("## 1. Previously completed and locked experiments\n")
    lines.append("- Primary Probe / Event-MIL / LEBRef five-fold subject-independent evaluation: complete.")
    lines.append("- Matched window-loss and occupancy controls: complete.")
    lines.append("- Mean / Max / normalized LSE / Smooth-max aggregation baselines: complete.")
    lines.append("- Five-point temperature sensitivity (0.05/0.07/0.10/0.15/0.20): complete.")
    lines.append("- Complete negative trial vs 10/31/100-window sub-bags: complete.")
    lines.append("- Positive/negative bag-side pooling factorial: complete.")
    lines.append("- Residual-MLP capacity control: complete.")
    lines.append("- Three-seed stability: complete.")
    lines.append("- Operating-point analysis: complete.")
    lines.append("- Activity-level analysis: complete.")
    lines.append("- V18 paired sign-flip tests + Holm correction: complete.\n")

    mainp = known["main_signflip_holm"]
    lines.append("### Locked main statistical results")
    lines.append(f"- SmartFallMM mean duty: Holm p = {mainp['smartfallmm']['mean_duty_p']}; mean FA/h: p = {mainp['smartfallmm']['mean_fah_p']}.")
    lines.append(f"- UMAFall mean duty: Holm p = {mainp['umafall']['mean_duty_p']}; mean FA/h: p = {mainp['umafall']['mean_fah_p']}.\n")

    lines.append("## 2. New final diagnostic: segmented negative-bag gradients\n")
    if grad:
        primary = [r for r in grad if r.get("seed") == "20260724" and r.get("setting") == "primary"]
        for dataset in ("smartfallmm", "umafall"):
            for method in ("event_mil", "lebref"):
                rows = [r for r in primary if r.get("dataset") == dataset and r.get("method") == method]
                if rows:
                    parts = ", ".join(f"{r['region']}={100*float(r['mean_abs_gradient_share']):.1f}%" for r in rows)
                    lines.append(f"- {dataset} / {method}: {parts}.")
        lines.append("\nInterpretation boundary: these are direct diagnostic gradient shares under the fixed trained head and negative-bag loss; they are not causal effect sizes.\n")
    else:
        lines.append("**MISSING:** negative-gradient output not found. The final closure is not complete.\n")

    lines.append("## 3. New final diagnostic: SmartFallMM age-group false-alarm burden\n")
    if age:
        for r in age:
            if r.get("seed") == "20260724":
                lines.append(f"- {r['age_group']}: n={r['n_subjects']}, Probe duty={100*float(r['probe_mean_duty']):.2f}%, LEBRef duty={100*float(r['lebref_mean_duty']):.2f}%, Δ={100*float(r['mean_delta_duty']):+.2f} pp [{100*float(r['delta_ci_low']):+.2f}, {100*float(r['delta_ci_high']):+.2f}].")
        if age_did:
            r = next((x for x in age_did if x.get("seed") == "20260724"), age_did[0])
            lines.append(f"- Older-minus-younger difference in mean Δ duty: {100*float(r['difference_of_mean_differences']):+.2f} pp [{100*float(r['ci_low']):+.2f}, {100*float(r['ci_high']):+.2f}].")
        lines.append("\nInterpretation boundary: observational subgroup analysis only; do not claim that age causes the improvement.\n")
    else:
        lines.append("**MISSING:** age-group output not found.\n")

    lines.append("## 4. New final diagnostic: motion-intensity heterogeneity\n")
    if intensity:
        for dataset in ("smartfallmm", "umafall"):
            rows = [r for r in intensity if r.get("dataset") == dataset and r.get("seed") == "20260724"]
            if rows:
                lines.append(f"### {dataset}")
                for r in rows:
                    lines.append(f"- {r['intensity_tercile']}: n={r['n_subjects']}, Δ duty={100*float(r['mean_delta_duty']):+.2f} pp [{100*float(r['delta_ci_low']):+.2f}, {100*float(r['delta_ci_high']):+.2f}].")
        for r in intensity_corr:
            if r.get("seed") == "20260724":
                lines.append(f"- {r['dataset']} Spearman intensity vs Δ duty: rho={float(r['spearman_rho_intensity_vs_delta_duty']):.3f}, p={float(r['spearman_p']):.4g}.")
        lines.append("\nInterpretation boundary: the intensity variable is an accelerometer-derived proxy, not a clinical activity-intensity label.\n")
    else:
        lines.append("**MISSING:** motion-intensity output not found.\n")

    lines.append("## 5. New final figure support: true two-dimensional precision–recall data\n")
    if curve:
        lines.append("- Alarm-segment precision is defined as detected observable fall events / (detected observable fall events + fully non-overlapping false-alarm segments).")
        lines.append("- The package outputs both alarm-segment PR curves and the deployment-oriented Recall–Duty curves. This satisfies the teacher's request for a 2D PR view without discarding the paper's duty-centered operating analysis.\n")
    else:
        lines.append("**MISSING:** PR/operating-curve output not found.\n")

    missing = []
    for name, ok in [("negative gradient", bool(grad)), ("age", bool(age)), ("motion intensity", bool(intensity)), ("PR curves", bool(curve))]:
        if not ok:
            missing.append(name)
    lines.append("## 6. Final closure decision\n")
    if not missing:
        lines.append("**FINAL_EXPERIMENTAL_CLOSURE = PASSED.** All teacher-requested experimental/diagnostic gaps targeted by V19 have an output. No further training experiment is planned.")
    else:
        lines.append("**FINAL_EXPERIMENTAL_CLOSURE = INCOMPLETE.** Missing: " + ", ".join(missing) + ".")
    lines.append("\nAfter this point, work should move to manuscript wording, table/figure integration, and visual QA rather than new model training.\n")

    report = root / "FINAL_EXPERIMENTAL_CLOSURE_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    atomic_json(root / "FINAL_EXPERIMENTAL_CLOSURE_STATUS.json", {"passed": not missing, "missing": missing})
    print(report)


if __name__ == "__main__":
    main()

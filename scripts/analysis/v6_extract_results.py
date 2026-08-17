from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract concise V6 results.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "v6_results",
    )
    return parser.parse_args()


def show_summary(label: str, row: dict) -> None:
    print(f"\n[{label}]")
    for key in (
        "all_event_recall",
        "observable_event_recall",
        "mean_subject_false_alarms_per_hour",
        "pooled_false_alarms_per_hour",
        "mean_subject_false_alarm_duty_cycle",
        "pooled_false_alarm_duty_cycle",
        "worst_subject_false_alarm_duty_cycle",
        "cvar80_subject_false_alarm_duty_cycle",
        "event_external_alarm_seconds_per_hour",
        "median_alarm_segment_seconds",
        "p90_alarm_segment_seconds",
        "maximum_alarm_segment_seconds",
        "proxy_median_delay_seconds",
        "proxy_p90_delay_seconds",
    ):
        value = row.get(key)
        print(f"{key} = {value:.6f}" if isinstance(value, float) else f"{key} = {value}")


def main() -> None:
    args = parse_args()
    for dataset in ("smartfallmm", "umafall"):
        print(f"\n===== {dataset.upper()} =====")
        audit_path = (
            args.results_root
            / dataset
            / "locked_alarm_duty_audit"
            / "locked_alarm_duty_audit.json"
        )
        if not audit_path.is_file():
            print("locked audit: MISSING")
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        show_summary("frozen_weda_target_calibrated", audit["frozen_weda_primary_oof"])
        show_summary("standard_head_tune", audit["standard_head_primary_oof"])
        duty_ci = audit["paired_primary_oof_subject_bootstrap"][
            "mean_subject_false_alarm_duty_cycle"
        ]
        print(
            "\nlocked_head_minus_frozen_mean_subject_false_duty "
            f"difference={duty_ci['mean_difference']:.6f} "
            f"95%CI=[{duty_ci['ci95_lower']:.6f}, {duty_ci['ci95_upper']:.6f}]"
        )
        esra_path = args.results_root / dataset / "esra_ablation" / "esra_results.json"
        if not esra_path.is_file():
            print("\nESRA ablation: NOT RUN (audit-only result is valid)")
            continue
        esra = json.loads(esra_path.read_text(encoding="utf-8"))
        for variant, payload in esra["variants"].items():
            if payload["primary_oof"] is None:
                print(f"\n[{variant}] incomplete")
                continue
            show_summary(variant, payload["primary_oof"])
            ci = payload["paired_vs_standard_head_bootstrap"][
                "mean_subject_false_alarm_duty_cycle"
            ]
            print(
                f"{variant}_minus_standard_mean_subject_false_duty "
                f"difference={ci['mean_difference']:.6f} "
                f"95%CI=[{ci['ci95_lower']:.6f}, {ci['ci95_upper']:.6f}]"
            )
    print("\nFALL_ESRA_V6_RESULT_EXTRACTION_OK")


if __name__ == "__main__":
    main()

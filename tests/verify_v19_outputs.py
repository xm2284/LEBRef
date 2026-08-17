from __future__ import annotations
import argparse, json
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--output-dir', required=True); a=p.parse_args()
r=Path(a.output_dir)
required=[
 'final_new_analyses/negative_gradient/negative_gradient_subject_equal_summary.csv',
 'final_new_analyses/age_group/smartfallmm_age_group_summary.csv',
 'final_new_analyses/motion_intensity/motion_intensity_tercile_summary.csv',
 'final_new_analyses/pr_operating_curves/alarm_segment_pr_and_burden_curves.csv',
 'FINAL_EXPERIMENTAL_CLOSURE_REPORT.md','FINAL_EXPERIMENTAL_CLOSURE_STATUS.json'
]
missing=[x for x in required if not (r/x).is_file() or (r/x).stat().st_size == 0]
if missing: raise SystemExit('OUTPUT_VERIFY_FAILED missing/empty:\n'+'\n'.join(missing))
status=json.loads((r/'FINAL_EXPERIMENTAL_CLOSURE_STATUS.json').read_text())
if not status.get('passed'): raise SystemExit(f'OUTPUT_VERIFY_FAILED status={status}')
print('FINAL_OUTPUT_VERIFY_PASSED')

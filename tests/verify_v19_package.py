from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "README.md", "FINAL_PROTOCOL_V19.md", "run_final_closure.sh", "requirements_no_torch.txt",
    "scripts/common.py", "scripts/negative_gradient_diagnostic.py", "scripts/age_group_analysis.py",
    "scripts/motion_intensity_analysis.py", "scripts/pr_operating_curve_analysis.py", "scripts/make_final_report.py",
    "scripts/archive_results.sh", "reference_results/KNOWN_RESULTS.json",
    "embedded_sources/linear_event_core_v9.py", "embedded_sources/run_experiment_v9.py",
    "embedded_sources/fall_esra_head_v6_20260724.zip",
    "embedded_sources/fall_lea_aggregation_baselines_v17_exactv9_20260810.zip",
    "embedded_sources/fall_lea_v18_statistical_closure_20260810.zip",
    "embedded_results/05_实验结果.zip",
    "embedded_results/fall_lea_mechanism_decomposition_v13_results.tar.gz",
    "embedded_results/fall_lea_capacity_control_v14_results.tar.gz",
]


def sha256(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

missing=[x for x in REQUIRED if not (ROOT/x).exists()]
if missing:
    raise SystemExit("PACKAGE_VERIFY_FAILED missing:\n"+"\n".join(missing))
known=json.loads((ROOT/"reference_results/KNOWN_RESULTS.json").read_text(encoding='utf-8'))
assert known["protocol"]["smoothmax_temperature_locked"] == 0.10
assert known["protocol"]["temperature_sweep"] == [0.05,0.07,0.10,0.15,0.20]
assert known["protocol"]["trainable_parameters"] == 513
manifest={"state":"ok","files":{x:sha256(ROOT/x) for x in REQUIRED}}
(ROOT/"PACKAGE_MANIFEST.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print("PACKAGE_VERIFY_PASSED")

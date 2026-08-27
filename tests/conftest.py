from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make the curated core modules importable by their plain module names
# (esra_core, v6_run_experiment, run_experiment alias), matching how the V6
# runner is loaded at runtime: V6_CODE_DIR defaults to REPO_ROOT/src/core.
CORE = str((REPO_ROOT / "src" / "core").resolve())
if CORE not in sys.path:
    sys.path.insert(0, CORE)

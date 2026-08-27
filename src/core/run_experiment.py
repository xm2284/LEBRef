"""Explicit compatibility alias for the locked V6 experiment runner.

The curated public repository exposes the locked V6 runner as
``v6_run_experiment.py`` (renamed so its version is explicit). Two runtime
entry points still import the historical module name ``run_experiment``:

- ``scripts/reproduce/run_experiment_v9.py`` (``load_v6``)
- ``scripts/analysis/common.py`` (``load_runtime``)

This file is that compatibility alias: it re-exports the single canonical
implementation (``v6_run_experiment``) without duplicating any logic. It is
intentionally tiny and contains no scientific code, seeds, thresholds, or
result values.
"""

from v6_run_experiment import *  # noqa: F401,F403
from v6_run_experiment import (  # noqa: F401
    DATASET_CONFIG,
    PACKAGE_DIR,
    VARIANTS,
    file_sha256,
    legacy_integrity_check,
    load_legacy_head,
    load_runtime,
    metric_summary,
    parse_args,
    resolve_args,
    run_esra,
    run_locked_audit,
    split_subjects,
    training_config,
)

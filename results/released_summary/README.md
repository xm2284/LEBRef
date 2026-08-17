# Released summary results

This directory contains small JSON/CSV/Markdown files that support inspection of the paper results without committing raw datasets, full feature caches, or trained checkpoints.

Key files:

- `KNOWN_RESULTS.json`: locked numerical anchors for main tables and comparisons.
- `v9_main/*_linear_event_results.json`: main V9 output summaries when available.
- `v19_closure/*`: no-retraining diagnostic summaries from the final V19 closure.

Large result directories and trained heads should be distributed as external artifacts, not committed to Git.

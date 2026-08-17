# Reproducibility audit summary

This public summary records the reproducibility status of the curated LEBRef
repository. Internal machine paths, manuscript drafts, teacher comments, and
large local artifacts are intentionally omitted.

## Scope

The repository keeps the code needed to reproduce and audit the experimental
pipeline for:

- frozen-representation event detection experiments;
- locked subject-independent evaluation on SmartFallMM and UMAFall;
- linear event-bag refinement experiments;
- V19 closure diagnostics, including age-group analysis, motion-intensity
  analysis, negative-bag gradient diagnostics, and alarm-segment PR/burden
  curves.

## Included

- Core Python source code under `src/`.
- Reproduction and analysis scripts under `scripts/`.
- Locked protocol notes under `docs/protocols/`.
- Small released summary tables under `results/released_summary/`.
- Test and verification scripts under `tests/`.
- Environment placeholders in `.env.example`.

## Excluded

The following are intentionally not committed:

- raw public datasets;
- feature caches;
- pretrained encoder checkpoints;
- trained downstream heads;
- manuscript PDFs, teacher comments, and private revision notes;
- local temporary outputs and cloud-compile archives.

See `docs/DATA_AND_ARTIFACTS.md` for the expected artifact layout.

## Current audit decision

The staged repository is suitable as a GitHub reproducibility skeleton. It is
not intended to include all historical V1--V19 intermediate folders. Instead,
it preserves the reusable source code, locked protocols, final diagnostic
scripts, and small summary results needed for readers to understand and rerun
the final experiments when the external datasets and checkpoints are provided.


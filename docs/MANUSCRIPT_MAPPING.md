# Manuscript-to-repository mapping

This note maps the Pervasive and Mobile Computing manuscript package to the
public repository contents. It is intended to help readers locate the released
summary files behind the main and supplementary analyses. Figure numbers below
follow the compiled manuscript PDFs; source figure filenames are listed only to
avoid confusion when cross-checking the LaTeX package.

## Main manuscript

| Manuscript item | Role in the paper | Repository materials |
|---|---|---|
| Figure 1 (`Figure_1.pdf`) | Method overview for LEBRef and the supervision--deployment mismatch | Conceptual schematic; no raw experimental data required |
| Figure 2 (`Figure_3.pdf`) | Primary subject-independent results: observable-event recall, false-alarm duty, and FA/h | `results/released_summary/KNOWN_RESULTS.json`; `results/released_summary/v9_main/*_linear_event_results.json` |
| Figure 3 (`Figure_4.pdf`) | Complete-negative-trial activation diagnostic for LEBRef relative to Event-MIL | `results/released_summary/v19_closure/negative_gradient_subject_equal_summary.csv` and V9 released summaries |
| Figure 4 (`Figure_7.pdf`) | Alarm-segment precision--recall and recall-target duty behavior | `results/released_summary/v19_closure/alarm_segment_pr_and_burden_curves.csv` |
| Table 1 and main result tables | Dataset scale, primary matched-protocol results, and component effects | `results/released_summary/KNOWN_RESULTS.json`; `results/released_summary/v9_main/` |

## Supplementary material

| Supplementary item | Role in the paper | Repository materials |
|---|---|---|
| Supplementary Figure 1 (`Figure_2.pdf`) | Representative fall and non-fall alarm timelines | Recreated from locked evaluation outputs; full local traces are not committed |
| Supplementary Figure 2 (`Figure_5.pdf`) | Stagewise alarm-burden changes | `results/released_summary/v9_main/`; `results/released_summary/v19_closure/FINAL_EXPERIMENTAL_CLOSURE_REPORT.md` |
| Supplementary Figure 3 (`Figure_6.pdf`) | Subject-level false-alarm-duty heterogeneity | `results/released_summary/v19_closure/smartfallmm_age_group_summary.csv`; `results/released_summary/v19_closure/motion_intensity_tercile_summary.csv`; `results/released_summary/v19_closure/motion_intensity_correlation.csv` |
| Supplementary tables | Aggregation sensitivity, temperature sweep, seed robustness, negative-bag controls, bag-side controls, operating targets, and activity-level changes | `results/released_summary/KNOWN_RESULTS.json`; `results/released_summary/v9_main/`; `results/released_summary/v19_closure/` |

## Notes on unavailable large files

Full feature caches, pretrained checkpoints, trained downstream heads, and raw
datasets are intentionally excluded from Git. The expected local artifact layout
is documented in `docs/DATA_AND_ARTIFACTS.md`.

# LEA causal-control suite v10 protocol

Status: locked before any v10 test-fold result is viewed.

## Question

Which part of the v9 Linear Event Adapter result is attributable to the Event-MIL supervision unit, Duty regularization, the proximal anchor, and variable negative-bag length when all new controls use the same 513 parameters, probe initialization, optimizer budget, folds, seeds, validation rule and alarm evaluator?

## Invariants

- Datasets: SmartFallMM and corrected UMAFall wrist subsets.
- Five fixed subject-independent folds from v9.
- Seeds: 20260724, 20260725 and 20260726.
- Frozen cached UniMTS features and training-fold StandardScaler.
- One 512-to-1 affine map: 513 parameters.
- AdamW, 15 epochs, 16 steps per epoch, learning rate 3e-4, weight decay 1e-4 and cosine annealing.
- Eight positive and eight negative bags per step.
- Candidate and threshold selection on the validation fold only.
- Threshold eligibility requires validation observable-event recall >= 0.90.
- Duration-aware validation ordering is identical to v9.
- No test-fold selection, tuning or candidate addition.

## Locked controls

1. `matched_window_anchor`: label every window in the sampled positive event bags positive and every window in the same sampled negative trial bags negative. Keep AdamW and the 0.01 proximal anchor. This is the matched-window control for v9 Event-MIL.
2. `matched_window_duty`: the same window loss plus Duty weight selected from {0.5, 1.0}. This tests whether Duty behaves differently without MIL aggregation.
3. `event_mil_no_anchor`: v9 Event-MIL with anchor weight 0. This isolates the proximal anchor.
4. `event_mil_fixed_negative`: v9 Event-MIL with each sampled negative bag replaced by a random contiguous 31-window sub-bag. This tests the fixed-length negative bag boundary.

The existing v9 `linear_event_mil` and final `linear_event_duty` records are reference methods and are not retrained.

## Interpretation rules

- Event-MIL attribution requires comparing v9 Event-MIL with `matched_window_anchor` under paired subject bootstrap, not comparing Event-MIL only with the L-BFGS probe.
- Duty attribution is described separately within the matched-window branch and the v9 Event-MIL branch.
- The no-anchor and fixed-negative variants are diagnostic controls, not candidate replacements for LEA.
- A confidence interval crossing zero is inconclusive, not evidence of equivalence.
- Recall, mean and pooled FA/h, mean and pooled Duty, tail Duty and proxy delay must all be retained.
- The experiment remains developmental because both datasets informed earlier method development.

## Success questions

The suite is informative regardless of direction. The manuscript may strengthen a causal claim only if the matched Event-MIL comparison reduces burden without a statistically clear recall disadvantage on both datasets. Otherwise the manuscript must retain the narrower staged-association wording.

No CVaR, temporal-consistency network, larger alpha grid, extra random seed or post hoc threshold rule may be added to v10.

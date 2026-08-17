# Fall ESRA-Head V6 Frozen Protocol

Date locked: 2026-07-24

## Status and scope

This protocol is a post hoc diagnostic and development benchmark created after discovering that the locked FA/h evaluator counted alarm segments but did not measure alarm duration. SmartFallMM and UMAFall have already been inspected and are not untouched confirmatory datasets. No result from V6 may be described as pristine external confirmation.

## Stage A: mandatory locked duration audit

- Reuse the original five subject folds, three seeds, saved head weights, validation-selected thresholds, window scores, event proxy, grace period, K=1, and refractory=2 seconds.
- Do not retrain a model or select a new threshold.
- Recompute the original recall, FA/h, and delay endpoints and require absolute agreement with the locked result JSON within 1e-6.
- Define the monitoring interval as the interval from the first causal score time to trial end.
- Split an alarm crossing the proxy event interval by overlap duration. Only alarm seconds outside the clipped proxy-plus-grace interval count as event-external alarm seconds.
- Report total alarm duty, false-alarm duty, event-external alarm seconds per hour, alarm-duration quantiles, and subject mean, worst, and CVaR80 burden.
- Use the primary seed for OOF point estimates and 10,000 paired subject-bootstrap replicates. Other seeds are sensitivity runs.

Stage A is sufficient to proceed with a diagnostic/trade-off manuscript. It does not validate a new learning method.

## Stage B: optional ESRA-Head development benchmark

The UniMTS encoder remains frozen. Every learned method uses the same 2,050-parameter head:

`LayerNorm(512) -> Dropout(0.1) -> Linear(512, 2)`

All heads start from the same WEDA head state. For each outer test fold, the next fold is validation and the remaining three folds are training. Test subjects are never used for checkpoint, loss-weight, or threshold selection.

### Loss

`L = L_event + alpha * L_duty + beta * L_CVaR80 + gamma * L_temporal`

- `L_event`: smooth-max multiple-instance loss over observable fall-event bags and nonfall trial bags.
- `L_duty`: mean soft fall probability over non-event windows, including event-external windows in fall trials.
- `L_CVaR80`: top-20% mean subject soft background burden in each sampled subject batch.
- `L_temporal`: total variation of adjacent window probabilities.

The temporal term is never evaluated without the duty term because smooth but continuously high alarms are undesirable.

### Ablations and validation-only grids

- `event_mil`: alpha=0, beta=0, gamma=0.
- `event_duty`: alpha in {0.5, 1.0}.
- `event_duty_cvar`: alpha in {0.5, 1.0}, beta in {0.5, 1.0}.
- `esra_full`: alpha in {0.5, 1.0}, beta in {0.5, 1.0}, gamma in {0.01, 0.05}.

Each candidate is trained for at most 15 epochs with patience 4, AdamW learning rate 1e-3, weight decay 1e-4, and a cosine schedule. Validation event-level average precision selects the checkpoint.

For each candidate, thresholds 0.01 through 0.99 are evaluated on validation subjects. Eligible thresholds require observable event recall >= 0.90. Selection is lexicographic: mean-subject false-alarm duty, pooled false-alarm duty, mean-subject FA/h, CVaR80 false-alarm duty, worst-subject false-alarm duty, median delay, then higher threshold. Candidate selection uses the same validation ordering.

### Test endpoints

Co-primary descriptive endpoints:

1. observable event recall;
2. mean-subject false-alarm duty cycle.

Secondary endpoints include all-event recall, FA/h, pooled and tail duty, event-external alarm seconds per hour, median/P90/maximum alarm duration, and median/P90 delay.

Comparisons use primary-seed OOF subject contributions and 10,000 paired subject-bootstrap replicates. SmartFallMM and UMAFall are reported separately. No recall-preservation, equivalence, or non-inferiority claim is allowed without a separately justified margin and compatible confidence interval.

## Interpretation gate

- If locked Head Tune lowers alarm counts and duration burden, the existing trade-off conclusion is strengthened.
- If counts fall but duration burden does not, the paper must distinguish alarm-count reduction from alarm-burden reduction.
- If duration burden rises, the locked method is not a deployment improvement under V6; Stage B may be reported as a corrective development experiment if its ablations support the mechanism.


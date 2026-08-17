# FINAL PROTOCOL V19 — LEBRef Experimental Closure

## Purpose
This package is the **final experimental closure** for the LEBRef manuscript. It does **not retrain** experiments that have already been completed and audited. It imports/archives the completed experiment evidence and runs only the remaining teacher-requested post-hoc diagnostics:

1. direct segmented negative-bag gradient attribution;
2. SmartFallMM age-group false-alarm analysis;
3. accelerometer-derived motion-intensity heterogeneity analysis;
4. two-dimensional alarm-segment precision–recall curves while retaining Recall–Duty curves.

After this run, no further model-training experiment is planned unless a new reviewer explicitly requests one.

## Locked upstream evidence
- V9: primary Probe / Event-MIL / LEBRef five-fold subject-independent protocol.
- V10: causal/matched window controls.
- V11: five-point smooth-max temperature sensitivity.
- V12: operating-point audit.
- V13: negative-bag range and bag-side decomposition.
- V14: residual-MLP output-capacity control.
- V17: capacity-matched Mean / Max / normalized-LSE / Smooth-max aggregation baselines with exact V9 parity.
- V18: paired sign-flip p-values + Holm correction and temperature audit.

## No-training rule
The V19 runner must never call V9/V10/V13/V14/V17 training entry points. Existing model heads/checkpoints and feature caches are read only.

## New analysis A — segmented negative-bag gradient attribution
For each complete non-fall test trial under the selected Event-MIL and LEBRef heads, compute the negative bag loss

`-log(1 - smoothmax(p_t))`

and its gradient magnitude with respect to each window logit. The primary region definition is fixed before examining outputs:

- hard trigger: top 1% of trial probabilities, minimum 1 window;
- sustained activation: remaining windows with `p_t >= 0.5 * selected_threshold` in runs of at least 5 windows (0.5 s at 0.1-s score stride);
- remaining background: all other windows.

Primary summary: subject-equally weighted fraction of total absolute gradient mass in each region, with 10,000 subject bootstrap replicates. The package can also run a small definition-sensitivity grid.

## New analysis B — age-group false-alarm burden
SmartFallMM subject IDs encode `young_` and `old_`. Compare Probe vs LEBRef subject-level false-alarm duty within younger and older groups. Report paired mean changes and 95% subject-bootstrap intervals. Do not compare fall recall across age groups because fall trials are contributed by the younger cohort. Treat this as an observational subgroup analysis, not a causal age effect.

## New analysis C — motion-intensity heterogeneity
For each non-fall trial, compute dynamic acceleration vector-magnitude RMS after subtracting the trial median magnitude. Aggregate per subject by the median across their non-fall trials. Define dataset-specific subject tertiles from this intensity proxy before using alarm outcomes. Report Probe and LEBRef duty and paired change by intensity tertile, plus Spearman correlation between subject intensity and duty change. This is an accelerometer-derived motion proxy, not a clinical intensity label.

## New analysis D — true 2D PR + deployment curves
Using the already selected primary-seed fold-specific models, sweep common thresholds 0.01–0.99 without reselecting models. Define:

- observable-event recall = detected observable fall events / observable fall events;
- alarm-segment precision = detected observable fall events / (detected observable fall events + fully non-overlapping false-alarm segments).

Output both:
- alarm-segment Precision–Recall curves;
- Recall–Duty operating curves.

This satisfies the teacher's request for a 2D PR view while retaining the deployment-oriented burden analysis central to LEBRef.

## Statistical rules
- bootstrap iterations: 10,000 by default;
- subject is the resampling unit where a subject-level estimand is reported;
- V18 p-values remain the inferential results for the primary method comparison;
- no new model selection or threshold selection is performed in V19;
- seeds are sensitivity repetitions and are not treated as independent subjects.

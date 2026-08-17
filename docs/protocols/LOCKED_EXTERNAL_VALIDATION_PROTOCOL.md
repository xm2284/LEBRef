# Locked UMAFall and SmartFallMM External Validation Protocol

Protocol version: `locked_umafall_smartfallmm_external_v1_20260723`

## Scientific role

UMAFall and SmartFallMM are the second and third locked external tests of the
15 WEDA-FALL-trained Acc-only UniMTS checkpoints. HIFD remains the first locked
external test and its negative result must remain in the paper. None of these
external datasets may select checkpoints, channels, axes, thresholds, SRRC
parameters, resampling rules, units, event windows, or exclusions.

## Fixed model and controller inputs

- Models: all 5 WEDA subject-independent folds x 3 seeds.
- Input: three-axis wrist acceleration; model channels 4-6 are zero-filled.
- Model rate: 50 Hz.
- Window: causal 200 samples (4.0 s), stride 5 samples (0.1 s).
- Event proxy: maximum acceleration magnitude within each published fall trial.
- Proxy interval: 1.5 s before to 1.5 s after the impact proxy, plus 0.5 s grace.
- Controllers: `fixed_0.5`, `max_f1`, `sensitivity_constrained`, and SRRC from
  the unchanged WEDA-only global lock.

## UMAFall lock

- Use only `UMAFall_Dataset_corrected_version.zip`.
- Use all 746 corrected CSV files: 19 subjects, 208 falls, and 538 ADLs.
- Select header-declared `WRIST`, `Sensor Type=0`; corrected files must map it
  to sensor ID 3.
- Treat acceleration as g and multiply by 9.80665 to obtain m/s2.
- Use recorded millisecond timestamps and linear interpolation to 50 Hz.
- No corrected trial is excluded; all have at least four seconds of valid time.

## SmartFallMM lock

- Use smartwatch accelerometer CSVs from both Young and Old groups.
- Use all observed watch subjects because this is an accelerometer-only test;
  completeness of phone, Meta, skeleton, or gyroscope modalities is irrelevant.
- Young A10-A14 are falls. Young A01-A09 and Old A01-A08 are non-falls.
- Parse the final three columns as x, y, z for both supported four- and
  five-column files.
- Exclude before inference only files with fewer than 128 valid samples.
- Frozen retained set: 57 subjects, 2,314 trials, 658 falls, 1,656 non-falls;
  93 of 2,407 source files are excluded by the length rule.
- Use the documented nominal 32 Hz sample index and linear interpolation to
  50 Hz. Do not select an alternative time interpretation after seeing scores.
- Treat watch values as m/s2 without conversion. This is a documented protocol
  assumption supported by the approximately 9.81 stationary norm audit; report
  the assumption as a limitation because the repository README omits units.

## Metrics and interpretation

Report impact-proxy event recall, pooled and mean-subject false alarms per hour,
worst-subject false alarms per hour, CVaR80 subject false alarms per hour,
proxy median/P90 delay, per-scenario results, and paired SRRC-minus-
`sensitivity_constrained` differences over the 15 checkpoints. P-values are
descriptive because checkpoints are not independent datasets.

Do not call the proxy outcomes manual-onset recall or clinical delay. Report all
three locked external datasets regardless of whether SRRC improves, ties, or
worsens. Any later unit, calibration, or domain-adaptation experiment must be
labelled exploratory and cannot replace these results.

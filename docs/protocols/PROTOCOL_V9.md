# Linear Event Adapter v9 Locked Protocol

## Objective

Test whether a standardized 513-parameter linear probe can be refined with
event-level MIL and non-event duty regularization while retaining its short
alarm segments and reducing alarm counts.

## Datasets and splits

- SmartFallMM and corrected UMAFall.
- Reuse the exact v6 five-fold subject locks.
- For outer fold `f`, fold `f` is test, fold `(f+1) mod 5` is validation, and
  the remaining folds are training.
- Primary seed: 20260724. Sensitivity seeds: 20260725 and 20260726.

## Frozen representation

Reuse the locked UniMTS feature caches. The encoder is never updated. Feature
standardization is fit only on training-subject windows in each outer fold.

## Methods

1. `linear_probe`: balanced Logistic Regression with C in
   `{0.01, 0.1, 1, 10}`.
2. `linear_event_mil`: initialize from the selected probe and optimize
   event-level MIL with a fixed proximal anchor of 0.01.
3. `linear_event_duty`: add mean non-event probability regularization with
   alpha in `{0.5, 1.0}`; select alpha on validation only.

All trainable heads contain one 512-to-1 affine map: 513 parameters. No CVaR
or temporal consistency term is used.

## Selection

- Candidate C, epoch, alpha and operating threshold are selected using the
  validation fold only.
- A threshold is eligible only when validation observable-event recall is at
  least 0.90.
- Among eligible points, use the locked v6 duration-aware burden ordering:
  mean subject false-alarm duty, pooled duty, CVaR80 duty, worst duty, delay,
  then threshold.
- The outer test fold is evaluated once after selection.

## Primary success gate

The method is considered promising only if, on both datasets:

1. primary OOF observable recall is no more than 0.01 below the linear probe;
2. mean subject false-alarm duty is not higher than the linear probe;
3. pooled false alarms per hour is lower than the linear probe; and
4. the paired subject bootstrap does not show a clear duty disadvantage.

Failure of this gate ends model optimization. Test results must not be used to
add new C, alpha, anchor or threshold candidates to v9.

## Reporting

Report all folds, all seeds, all validation candidates, subject-level OOF
summaries, alarm duration endpoints, and paired 10,000-iteration subject
bootstrap comparisons against the linear probe, Standard Head and ESRA-Full.

This is developmental evidence because the two datasets have already
participated in method development. It is not an untouched external
confirmation.

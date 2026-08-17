# LEBRef Final Experimental Closure Report (V19)

**Run type:** no-retraining final closure. Existing training experiments are treated as locked; this run adds only the remaining diagnostics requested by the teachers.

## 1. Previously completed and locked experiments

- Primary Probe / Event-MIL / LEBRef five-fold subject-independent evaluation: complete.
- Matched window-loss and occupancy controls: complete.
- Mean / Max / normalized LSE / Smooth-max aggregation baselines: complete.
- Five-point temperature sensitivity (0.05/0.07/0.10/0.15/0.20): complete.
- Complete negative trial vs 10/31/100-window sub-bags: complete.
- Positive/negative bag-side pooling factorial: complete.
- Residual-MLP capacity control: complete.
- Three-seed stability: complete.
- Operating-point analysis: complete.
- Activity-level analysis: complete.
- V18 paired sign-flip tests + Holm correction: complete.

### Locked main statistical results
- SmartFallMM mean duty: Holm p = 0.0008; mean FA/h: p = 0.0003.
- UMAFall mean duty: Holm p = 0.0021; mean FA/h: p = 0.181.

## 2. New final diagnostic: segmented negative-bag gradients

- smartfallmm / event_mil: hard_trigger=37.2%, remaining_background=38.9%, sustained_activation=23.9%.
- smartfallmm / lebref: hard_trigger=37.6%, remaining_background=39.8%, sustained_activation=22.6%.
- umafall / event_mil: hard_trigger=42.6%, remaining_background=46.6%, sustained_activation=10.8%.
- umafall / lebref: hard_trigger=42.4%, remaining_background=47.2%, sustained_activation=10.4%.

Interpretation boundary: these are direct diagnostic gradient shares under the fixed trained head and negative-bag loss; they are not causal effect sizes.

## 3. New final diagnostic: SmartFallMM age-group false-alarm burden

- younger: n=32, Probe duty=7.41%, LEBRef duty=5.99%, Δ=-1.41 pp [-2.67, -0.40].
- older: n=25, Probe duty=1.23%, LEBRef duty=0.77%, Δ=-0.46 pp [-0.85, -0.16].
- Older-minus-younger difference in mean Δ duty: +0.96 pp [-0.13, +2.26].

Interpretation boundary: observational subgroup analysis only; do not claim that age causes the improvement.

## 4. New final diagnostic: motion-intensity heterogeneity

### smartfallmm
- low: n=19, Δ duty=-0.37 pp [-0.93, +0.01].
- middle: n=18, Δ duty=-0.68 pp [-1.27, -0.20].
- high: n=19, Δ duty=-1.19 pp [-2.53, -0.10].
### umafall
- low: n=7, Δ duty=-0.72 pp [-1.57, -0.04].
- middle: n=6, Δ duty=-0.43 pp [-0.83, -0.10].
- high: n=6, Δ duty=-2.67 pp [-5.18, -0.32].
- smartfallmm Spearman intensity vs Δ duty: rho=0.018, p=0.8977.
- umafall Spearman intensity vs Δ duty: rho=-0.392, p=0.09668.

Interpretation boundary: the intensity variable is an accelerometer-derived proxy, not a clinical activity-intensity label.

## 5. New final figure support: true two-dimensional precision–recall data

- Alarm-segment precision is defined as detected observable fall events / (detected observable fall events + fully non-overlapping false-alarm segments).
- The package outputs both alarm-segment PR curves and the deployment-oriented Recall–Duty curves. This satisfies the teacher's request for a 2D PR view without discarding the paper's duty-centered operating analysis.

## 6. Final closure decision

**FINAL_EXPERIMENTAL_CLOSURE = PASSED.** All teacher-requested experimental/diagnostic gaps targeted by V19 have an output. No further training experiment is planned.

After this point, work should move to manuscript wording, table/figure integration, and visual QA rather than new model training.

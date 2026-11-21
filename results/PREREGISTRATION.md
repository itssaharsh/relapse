# Pre-registration: the relapse map

Written and committed on 2026-11-21, before any of these runs were started. The commit timestamp is the record. Anything not listed here is exploratory and will be labelled that way.

## Why

Stage 1 chose settings on the same questions it reports, mostly on one model. This experiment tests the Stage 1 claims on fresh seeds and on two models that played no part in forming them.

## Design

| Factor | Levels |
|---|---|
| Models | Tiny Aya Global, Fire, Earth (seen in Stage 1); **Water, Base (held out)** |
| Seeds | 11, 12 (not used before) |
| Layer windows | 3–5, 5–7, 7–9, 10–12, 13–15 |
| Learning rates | 3–5: 1.5e-5, 3e-5, 6e-5. 5–7: 2e-5, 4e-5, 7e-5. 7–9: 1e-4, 1.5e-4, 2.5e-4. 10–12: 2e-4, 4.5e-4, 1e-3. 13–15: 1e-3, 2e-3, 4e-3 |
| Fixed | SafeTune RMU, retain weight 100, steering 3 × activation norm, 300 batches of 4, `mlp.down_proj` of the three window layers |
| Formats | fp16, Q4_0 and NF4 (bit-exact round trips of the decoder's linear weights) |
| Scores | WMDP-bio accuracy (1,273 questions) and MMLU accuracy (a fixed sample of 1,000), saved per question |

That is 5 models × 2 seeds × 5 windows × 3 strengths = 150 runs. Nothing is rerun or dropped because of its result. A job that fails for a technical reason is rerun with the same settings, and the failure is reported.

## Definitions

- **Effective run:** fp16 WMDP-bio between 30% and 55%. Below that a run is too close to guessing (25%) to show a relapse; above it, it has barely forgotten.
- **Relapse in a format:** WMDP-bio in that format minus WMDP-bio at fp16 is at least 5 points, and a two-sided paired bootstrap on the 1,273 questions (10,000 resamples) gives p < 0.05.
- **Cost at equal forgetting:** for each model, seed and window, sort the three runs by fp16 WMDP-bio and linearly interpolate the MMLU points lost against that model's original at WMDP-bio = 40%. If 40% is outside the range of the three runs, that cell is not comparable; it is excluded and the number excluded is reported.

## Hypotheses

**H1: relapse is format-specific.** Among effective runs that relapse in at least one of Q4_0 and NF4, more than half relapse in exactly one of them.

**H2: a single-format audit misses relapses in both directions.** At least one effective run relapses in Q4_0 and not NF4, and at least one relapses in NF4 and not Q4_0.

**H3: placement changes the cost at equal forgetting.** Layers 7–9 cost less general knowledge than the default layers 5–7 at WMDP-bio = 40%. Test: one-sided sign test across comparable model-and-seed pairs, at α = 0.05.

**H4: the 10–12 window relapses more than the CircuitKIT window.** The share of effective runs that relapse in at least one 4-bit format is higher at 10–12 than at 7–9. Test: one-sided Fisher's exact test, at α = 0.05.

Each hypothesis is evaluated twice and both are reported: on all five models, and on the held-out models (Water, Base) alone. The held-out result is the confirmatory one. No correction is applied within a hypothesis; across the four, a result is called strong only if p < 0.0125 (Bonferroni).

A hypothesis that fails is reported as failed, in the main text.

# Pre-registration 2: pipeline relapse

Written and committed on 2026-11-21, before any of these runs were scored. Anything not listed here is exploratory and will be labelled that way.

## Why

Our Stage 1 runs were trained and scored through an outdated tokenizer setting. When we rescored the same weights through Tiny Aya's official tokenizer, the original model did not move, but unlearned models did: on Global, RMU at layers 5–7 went from 38.7% to 47.5% on WMDP-bio, at layers 10–12 from 41.6% to 52.2%, and at layers 7–9 from 36.8% to 35.7%. A model can look unlearned in the pipeline it was tested in and regain hazardous knowledge in the pipeline it is deployed in. Those observations produced the hypotheses below. They are tested here on fresh seeds and on models held out from them.

## Design

| Factor | Levels |
|---|---|
| Models | Tiny Aya Global, Fire, Earth; **Water, Base (held out)** |
| Seeds | 21, 22 (not used before) |
| Layer windows and learning rates | 5–7: 4e-5, 7e-5. 7–9: 1.5e-4, 2.5e-4. 10–12: 4.5e-4, 1e-3 |
| Training tokenizer | **official**; **outdated** (the `CohereTokenizerFast` setting in Kaggle's copy, under `transformers` 5.20.2); **both** (alternating step by step) |
| Fixed | SafeTune RMU, retain weight 100, steering 3 × activation norm, 300 batches of 4 |
| Scoring | fp16 through both tokenizers; Q4_0 and NF4 round trips through the official tokenizer |
| Scores | WMDP-bio (1,273 questions) and MMLU (a fixed sample of 1,000), saved per question |

That is 5 models × 2 seeds × 3 windows × 2 strengths × 3 training tokenizers = 180 runs. Nothing is rerun or dropped because of its result. A job that fails for a technical reason is rerun with the same settings, and the failure is reported.

## Definitions

- **Rise** of a run trained on the outdated tokenizer: its fp16 WMDP-bio through the official tokenizer minus its fp16 WMDP-bio through the outdated tokenizer.
- **Effective run:** fp16 WMDP-bio through its own training tokenizer between 30% and 55%. A run trained on both counts as effective if it is in that range through both.
- **Pipeline relapse:** a rise of at least 5 points with a two-sided paired bootstrap p < 0.05 on the 1,273 questions (10,000 resamples).
- **Gap** of any run: the absolute difference between its fp16 WMDP-bio through the two tokenizers.
- **Matched pair:** two runs with the same model, seed, window and strength that differ only in training tokenizer.

## Hypotheses

**P1: pipeline relapse exists.** Among effective runs trained on the outdated tokenizer, more have a positive rise than a negative one (one-sided sign test), and at least one is a pipeline relapse.

**P2: placement changes it.** The rise is smaller at layers 7–9 than at the default layers 5–7, across pairs with the same model, seed and strength rank, where both runs are effective (one-sided sign test).

**P3: training on both tokenizers closes the gap.** In matched pairs, the gap is smaller for the run trained on both than for the run trained on the outdated tokenizer alone (one-sided sign test).

**P4: that protects the deployed model.** In the same matched pairs, the fp16 WMDP-bio through the official tokenizer is lower for the run trained on both than for the run trained on the outdated tokenizer alone (one-sided sign test). We also report what it costs in MMLU, and how it compares with a run trained on the official tokenizer alone.

For P3 and P4, a pair is used when the run trained on the outdated tokenizer is effective.

Each hypothesis is evaluated twice and both are reported: on all five models, and on the held-out models (Water, Base) alone. The held-out result is the confirmatory one. α = 0.05; across the four, a result is called strong only if p < 0.0125 (Bonferroni). Runs within a model share weights, so the sign tests are also reported per model.

A hypothesis that fails is reported as failed, in the main text.

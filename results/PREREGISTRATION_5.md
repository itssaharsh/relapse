# Pre-registration 5: does placement protect against the tokenizer?

Written and committed on 2026-11-22, before any of these runs were started.

## Why

In the tokenizer experiment, runs unlearned through the outdated tokenizer setting and scored through the official one rose by 10.9 to 18.2 points at layers 5–7 (SafeTune's default) and by −2.7 to +0.5 points at layers 7–9. Layers 3–5 forget the most at the same general-knowledge cost and are the window we release, but they were never trained through the outdated setting. We test them in that direction, the one where a model looks unlearned in the pipeline it was built in and regains knowledge in the official one.

## Design

| Factor | Levels |
|---|---|
| Models | Tiny Aya Global, Fire, Earth, Water, Base |
| Seeds | 51, 52 (not used before) |
| Windows and learning rates | 3–5: 3e-5, 6e-5. 5–7: 4e-5, 7e-5. 7–9: 1.5e-4, 2.5e-4 |
| Training tokenizer | outdated |
| Fixed | SafeTune RMU, retain weight 100, steering 3 × activation norm, 300 batches of 4 |
| Scoring | fp16 through both tokenizer settings, per question |

That is 60 runs. Definitions are those of pre-registration 2: the **rise** is WMDP-bio through the official tokenizer minus through the outdated one; a run is **effective** if its WMDP-bio through its training tokenizer is between 30% and 55%; a **pipeline relapse** is a rise of at least 5 points with paired p < 0.05. Tiny Aya Base fell below the effective range at every window in the earlier experiment; its runs are reported whatever range they land in.

## Hypotheses

**T1.** The rise is smaller at layers 3–5 than at layers 5–7, across pairs with the same model, seed and strength rank where both runs are effective (one-sided sign test, α = 0.05).

**T2.** At layers 3–5, fewer than one in five effective runs is a pipeline relapse.

**T3 (replication).** At layers 7–9, fewer than one in five effective runs is a pipeline relapse, and at layers 5–7 more than half are.

Nothing is rerun or dropped because of its result. A job that fails for a technical reason is rerun with the same settings, and the failure is reported. A hypothesis that fails is reported as failed, in the main text.

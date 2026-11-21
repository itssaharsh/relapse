# Pre-registration 3: where to place the edit

Written and committed on 2026-11-21, before CircuitKIT was re-run with the official tokenizer and before any of these runs started.

## Why

The first pre-registered test of placement (H3) could not be evaluated: no model and seed reached 40% WMDP-bio at both layer windows. Looking at those runs afterwards, with a different measure, layers 7–9 beat the default 5–7 in 9 of 10 model-and-seed pairs, and layers 3–5 beat both in 10 of 10. That analysis was exploratory. This experiment tests it on fresh seeds, with the measure fixed in advance. CircuitKIT's Stage 1 attribution was also computed through the outdated tokenizer, so we recompute it.

## Design

**Step 1: CircuitKIT.** For each of the five Tiny Aya models, run EAP-IG through the official tokenizer with the Stage 1 settings (64 short WMDP-bio questions, 5 integration steps). **The CircuitKIT window** of a model is the three consecutive layers within layers 1–12 with the largest summed absolute MLP attribution. This is the rule Stage 1 used.

**Step 2: unlearning.**

| Factor | Levels |
|---|---|
| Models | Tiny Aya Global, Fire, Earth, Water, Base |
| Seeds | 31, 32 (not used before) |
| Windows | 3–5; 5–7 (SafeTune default); 7–9 (the Stage 1 pick); the model's CircuitKIT window from step 1, if different |
| Learning rates | Windows starting at layer 4 or below: 1.5e-5, 3e-5, 6e-5, 1e-4, 2e-4. Starting at 5 or 6: 4e-5, 7e-5, 1.5e-4, 3e-4, 6e-4. Starting at 7 or above: 1.5e-4, 2.5e-4, 5e-4, 1e-3, 2e-3 |
| Fixed | SafeTune RMU, official tokenizer, retain weight 100, steering 3 × activation norm, 300 batches of 4 |
| Formats | fp16, Q4_0 and NF4 round trips |

Nothing is rerun or dropped because of its result. A job that fails for a technical reason is rerun with the same settings, and the failure is reported.

## Measure

**Best safe forgetting** of a window, for a given model, seed and format: the lowest WMDP-bio among its five runs that lose at most 3 MMLU points against that model's original in the same format. If no run qualifies, that pair is not comparable; it is excluded and the number excluded is reported.

## Hypotheses

Each is a one-sided sign test across the 10 model-and-seed pairs at fp16, α = 0.05; across the three, a result is called strong only if p < 0.0167 (Bonferroni).

**Q1.** Layers 7–9 have lower best safe forgetting than the default layers 5–7.

**Q2.** Layers 3–5 have lower best safe forgetting than the default layers 5–7.

**Q3.** Each model's re-derived CircuitKIT window has lower best safe forgetting than the default layers 5–7.

The same comparisons in Q4_0 and NF4 are reported as secondary. A hypothesis that fails is reported as failed, in the main text.

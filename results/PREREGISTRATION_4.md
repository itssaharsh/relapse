# Pre-registration 4: the 4-bit relapse on Tiny Aya Base

Written and committed on 2026-11-21, before any of the new runs below were started.

## Why

The pre-registered relapse map (150 runs, official tokenizer) found one place where a 4-bit file brought forgotten knowledge back: Tiny Aya Base, seed 12, layers 10–12. At all three strengths the Q4_0 round trip scored higher on WMDP-bio than the same weights at fp16 (+5.7, +4.8 and +6.0 points, each p < 0.001), while NF4 did not rise. This is the pattern our Stage 1 submission described, now seen through the correct tokenizer. One seed is not enough. This experiment asks whether it happens again, whether it is real in the files people download, where it comes from, and whether training on the file repairs it.

**What we had already seen when writing this:** the three seed-12 runs above, and the first rows of a real-file check of two of them that was still running (one real Q4_0 file had scored 52.9% and one real Q4_K_M file 45.4%). Nothing else below had been run.

## Part A: does it happen again?

| Factor | Levels |
|---|---|
| Model | Tiny Aya Base |
| Seeds | 41 to 48 (not used before) |
| Layers | 10–12 |
| Learning rates | 2e-4, 4.5e-4, 1e-3 (the same three as in the map) |
| Fixed | SafeTune RMU, official tokenizer, retain weight 100, steering 3 × activation norm, 300 batches of 4 |
| Formats | fp16, Q4_0 and NF4 round trips |

Definitions are unchanged from the first pre-registration: an **effective run** has fp16 WMDP-bio between 30% and 55%; a **relapse** in a format is a rise of at least 5 points over the same weights at fp16 with a two-sided paired bootstrap p < 0.05. A **relapsing seed** has at least one effective run that relapses.

**R1: the relapse happens again.** At least one of the eight fresh seeds is a relapsing seed in Q4_0. We report the share of relapsing seeds with an exact 95% interval.

**R2: it is specific to the format.** On each seed, average the Q4_0 change and the NF4 change (4-bit minus fp16) over its effective runs. The Q4_0 change is larger than the NF4 change on more seeds than not (one-sided sign test across the eight fresh seeds, α = 0.05).

**Other models (no hypothesis).** Global, Fire, Earth and Water are run at layers 10–12 on seeds 41 and 42 with learning rates 4.5e-4, 1e-3 and 2e-3, and every relapse is reported.

## Part B: is it real in the files people download?

Every relapsing run, from seed 12 and from the fresh seeds, is rebuilt with llama.cpp as real Q4_0 and Q4_K_M files and scored.

**R3.** The relapse appears in the real Q4_0 file (a rise of at least 5 points over fp16, p < 0.05). Q4_K_M is reported either way.

## Part C: where does it come from?

For each relapsing run we round one part of the model at a time to Q4_0 and score WMDP-bio: only the layers in front of the edited window, only the three edited matrices, only the other matrices inside the window, only the layers behind it.

**R4.** Rounding only the layers in front of the edit raises WMDP-bio by more than rounding only the three edited matrices. Reported per run with paired bootstrap intervals; with at least five relapsing runs, a one-sided sign test.

## Part D: does training on the file repair it?

For every relapsing run we train a **repaired run** with the same seed and learning rate, in which the model and its frozen reference are seen as their Q4_0 files on every step, for 600 batches (the Stage 1 repair setting). Three controls use the same seed and learning rate: the same file training with 300 batches, plain RMU with 600 batches, and RMU with only the three edited matrices rounded during training.

**R5a.** The repaired run does not relapse in Q4_0.

**R5b.** In Q4_0, the repaired run scores lower on WMDP-bio than the plain run it repairs (paired bootstrap on the same questions, p < 0.05).

**R5c.** In Q4_0, the repaired run's MMLU is within 3 points of the plain run's.

R5a to R5c are evaluated per relapsing run, and the number of runs where each holds is reported.

## Rules

Nothing is rerun or dropped because of its result. A job that fails for a technical reason is rerun with the same settings, and the failure is reported. If no fresh seed relapses, R1 is reported as failed, and Parts B to D are reported for seed 12 alone and labelled as one seed. A hypothesis that fails is reported as failed, in the main text.

## Amendment 1 (2026-11-21, written before any Part E run was started): the NF4 direction

Parts A to D follow a relapse that appears in Q4_0 and not in NF4. The map also contains a hint of the opposite: Tiny Aya Fire, layers 13–15, seed 12, learning rate 2e-3 rose by 4.2 points in NF4 (p = 0.002) and did not rise in Q4_0. That is under the 5-point line, so it does not count as a relapse.

**Part E.** Tiny Aya Fire, layers 13–15, learning rates 1e-3 and 2e-3, seeds 41 to 48, same fixed settings and formats as Part A.

**R6: a relapse in the other direction exists.** At least one fresh seed has an effective run that relapses in NF4 and does not relapse in Q4_0. We report the share of such seeds with an exact 95% interval.

If R1 and R6 both hold, a single-format audit can miss a relapse in either direction (the claim that hypothesis H2 of the first pre-registration failed to support).

## Amendment 2 (2026-11-21, before its run was started): an exact rerun of seed 12

The seed-12 runs in the map were trained on a T4 GPU. We train the same three runs again, with the same seed and settings, on a different GPU type (L4). RMU training is sensitive to tiny numerical differences, so this shows whether the relapse belongs to the seed or to one particular training run. It is reported separately and does not count towards R1.

## Amendment 3 (2026-11-21, before its runs were started): cause and repair of the NF4 relapse

Part E found one relapse in the NF4 direction (Tiny Aya Fire, layers 13–15, seed 44, learning rate 2e-3: +5.3 points in NF4, no rise in Q4_0). We apply Parts B, C and D to it, with NF4 in the place of Q4_0: real llama.cpp files are scored for reference; one part of the model at a time is rounded to NF4; and the repaired run is trained with the model and its frozen reference seen as their NF4 copies on every step for 600 batches, with plain RMU for 600 batches as the control. R5a to R5c are evaluated in NF4 for this run.

## Amendment 4 (2026-11-22, before its runs were started): the Water relapse

The other-models runs of Part A found one more relapse: Tiny Aya Water, layers 10–12, seed 42, learning rate 1e-3 (+7.7 points in NF4, no rise in Q4_0). It gets the treatment of amendment 3: real llama.cpp files for reference, one part of the model rounded to NF4 at a time, and a repaired run trained on the NF4 copy for 600 batches with plain RMU for 600 batches as the control. For both NF4 relapses (Fire and Water) we also train on the NF4 copy for 300 batches, the control that Part D already has for Q4_0.

## Amendment 5 (2026-11-22, written after the Base and Fire repairs were scored and before any Water repair result existed): choosing the dose

What we had seen: training on the file for 600 batches removed the relapse in every case tried, but in three of four it also cost more than 3 MMLU points or pushed the model below the effective range. Training for 300 batches met all three criteria on Base seed 43 and did not repair Base seed 12.

**Dose rule.** Of the two file-trained runs (300 and 600 batches), take the one with more batches whose MMLU in the relapsing format is within 3 points of the plain run's. The rule looks only at MMLU, never at the hazard questions. If neither qualifies, the repair is reported as failed for that run.

**R7.** With the dose rule, the repaired run meets R5a, R5b and R5c. On the Base and Fire runs this is a description of data we had already seen. The Water run is the test.

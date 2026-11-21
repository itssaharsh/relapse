# Relapse: forget the hazard, keep the rest, even in the 4-bit file

Small models like Cohere Labs' [Tiny Aya](https://huggingface.co/CohereLabs/tiny-aya-global) run offline, on phones and laptops, where no server can filter what they say. Any safety has to live in the model's weights, and in the compressed file people actually download. We used four Lexsi Labs tools to make Tiny Aya forget hazardous biology and checked whether it stays forgotten in those files.

## Correction (November 21, 2026)

**Our Stage 1 results contained a serious error, and we have withdrawn two of the claims in our submission.**

**What happened.** Stage 1 ran on Kaggle's copy of Tiny Aya. That copy predates a fix Cohere made on Hugging Face on September 30, 2026: under `transformers` v5, its tokenizer setting splits text differently from the way the model was trained. All 1,273 WMDP-bio prompts and all 1,000 MMLU prompts were affected, in both training and scoring. The model weights themselves are fine.

**How we confirmed it.** We scored the same weights, in the same job, with both tokenizer settings:

| Model | WMDP-bio, outdated setting | WMDP-bio, official tokenizer |
|---|---|---|
| Original | 63.3% | 63.3% |
| RMU at the default layers (5 to 7) | 38.7% | **47.5%** |
| RMU at the CircuitKIT layers (7 to 9) | 36.8% | **35.7%** |
| RMU at layers 10 to 12 | 41.6% | **52.2%** |

**What we withdraw:**

- **The relapse.** We reported that an edit at layers 10 to 12 looked unlearned at full precision and regained 7.4 points in a 4-bit file. With the official tokenizer that model barely forgets (52.2%) and its 4-bit files do not rise (+0.3 points in Q4_0, p = 0.74; +0.7 in Q4_K_M, p = 0.47).
- **The repair.** Training on the 4-bit file had no real relapse to repair, and that model scores 58.1% with the official tokenizer.
- **Everything built on those two:** that relapse is format-specific, and that it comes from rounding the layers in front of the edit.
- **Every other Stage 1 number** (the sweeps, extra seeds, languages, prompt and fine-tuning attacks, Tiny Aya Fire and Earth) was measured with the outdated setting. We do not rely on any of it until it is re-run.

The original write-up is kept, marked as superseded, in [`docs/archive/`](https://github.com/itssaharsh/relapse/tree/main/docs/archive), and its raw runs in [`results/raw/`](https://github.com/itssaharsh/relapse/tree/main/results/raw).

## What holds with the official tokenizer

We rescored the Stage 1 models in full precision, a real llama.cpp Q4_K_M file, and exact Q4_0 and NF4 round trips. WMDP-bio is hazardous biology (1,273 questions, lower is better, 25% is guessing). MMLU is general knowledge (1,000 questions, higher is better).

<!-- scores:start -->
| File (WMDP-bio / MMLU) | Original | Default layers | CircuitKIT layers |
|---|---|---|---|
| fp16 | 63.3% / 55.4% | 47.5% / 53.1% | 35.7% / 54.3% |
| Q4_K_M | 63.2% / 54.7% | 45.0% / 51.6% | 36.8% / 53.8% |
| Q4_0 | 62.5% / 55.3% | 43.0% / 53.1% | 34.1% / 52.5% |
| NF4 | 60.3% / 53.5% | 46.4% / 49.9% | 36.1% / 51.7% |
<!-- scores:end -->

**The CircuitKIT-layer model keeps its forgetting in every file.** It scores 34.1% to 36.8% across the four files, against 60.3% to 63.3% for the original.

**Head to head, it forgets more than the default layers, at no significant cost in general knowledge.** Same questions, AuditKIT's paired bootstrap (difference is CircuitKIT minus default):

<!-- recipes:start -->
| File | WMDP-bio: default / CircuitKIT | Difference (95% interval) | p | MMLU: default / CircuitKIT | Difference (95% interval) | p |
|---|---|---|---|---|---|---|
| fp16 | 47.5% / 35.6% | -11.9 (-15.1 to -8.8) | 0.001 | 53.1% / 54.2% | +1.1 (-0.8 to +3.1) | 0.282 |
| Q4_K_M | 45.0% / 36.8% | -8.2 (-11.4 to -4.9) | 0.001 | 51.7% / 53.8% | +2.1 (+0.0 to +4.3) | 0.062 |
| Q4_0 | 43.0% / 34.2% | -8.9 (-11.9 to -5.8) | 0.001 | 53.1% / 52.5% | -0.6 (-2.7 to +1.5) | 0.561 |
| NF4 | 46.4% / 36.1% | -10.4 (-13.4 to -7.2) | 0.001 | 49.9% / 51.6% | +1.7 (-0.3 to +3.7) | 0.080 |
<!-- recipes:end -->

It forgets 8.2 to 11.9 points more in every file (p = 0.001 in each). The general-knowledge difference is not significant in any file.

**Read this result with care.** It is one training run of each recipe, trained with the outdated tokenizer and scored with the official one. The settings were also chosen under the outdated tokenizer, which may have favoured or hurt either recipe.

**A finding from the mistake itself.** The original model scored the same under both tokenizer settings, but the unlearned models moved by as much as 10.5 points (default layers +8.8, CircuitKIT layers -1.2, layers 10 to 12 +10.5). Unlearning measured through one preprocessing pipeline may not hold through another, so an audit should pin the exact tokenizer it was trained and tested with.

## What the pre-registered test found

We [registered](https://github.com/itssaharsh/relapse/blob/main/results/PREREGISTRATION.md) a 150-run test before starting it, using the **official tokenizer**: five Tiny Aya models, five layer placements, three strengths and two fresh seeds, with Water and Base held out. Of the 150 runs, 109 forgot enough to count.

**107 of those 109 did not relapse in Q4_0 or NF4.** With the correct tokenizer, RMU forgetting almost always survived 4-bit compression. The two relapses are both Tiny Aya Base, seed 12, layers 10 to 12, in Q4_0 only (+5.7 and +6.0 points). The relapse we reported on Global in Stage 1 was caused by the tokenizer.

| Registered hypothesis | Result |
|---|---|
| H1: a relapse tends to appear in one format, not both | Holds, but on only 2 relapses |
| H2: a single-format audit misses relapses in both directions | **Failed.** Q4_0 only: 2. NF4 only: 0 |
| H3: layers 7 to 9 cost less general knowledge than 5 to 7 at equal forgetting | **Could not be tested.** No model and seed reached the 40% comparison point at both placements |
| H4: layers 10 to 12 relapse more than layers 7 to 9 | **Failed.** 2 of 12 against 0 of 30, p = 0.077 |

Full results: [results/map/summary.json](https://github.com/itssaharsh/relapse/blob/main/results/map/summary.json).

## What is running now

- A [pre-registered](https://github.com/itssaharsh/relapse/blob/main/results/PREREGISTRATION_4.md) follow-up of the one relapse the map found (Tiny Aya Base, layers 10 to 12, Q4_0): eight fresh seeds, real llama.cpp files, where the relapse comes from, and whether training on the file repairs it.
- A [pre-registered](https://github.com/itssaharsh/relapse/blob/main/results/PREREGISTRATION_2.md) 180-run test of **pipeline relapse**: does a model unlearned through one tokenizer setting regain knowledge when deployed through another, and does training on both prevent it?
- A [pre-registered](https://github.com/itssaharsh/relapse/blob/main/results/PREREGISTRATION_3.md) test of **where to place the edit**, with CircuitKIT re-run through the official tokenizer on all five models and fresh seeds.

Their analysis scripts were committed before their results were read. This page will be updated from them, including any hypothesis that fails.

## Method

**Unlearning.** [SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s RMU edits the MLP output projection of three consecutive layers. On hazardous text it pushes activations toward a fixed random vector; on ordinary text (Wikitext) it holds them to a frozen copy of the model. [CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT)'s EAP-IG attribution pointed us to layers 7 to 9 instead of SafeTune's default 5 to 7.

**Scoring.** Each question is scored from the next-token probabilities of A to D, so no judge model is involved. [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) compares models on the same questions with a paired bootstrap.

**Formats.** The Q4_K_M file is built with llama.cpp and read back for scoring. Q4_0 and NF4 are exact round trips of the decoder's linear weights, with the embedding table left at fp16.

## Limits

- One run per recipe in the rescored comparison, trained with the outdated tokenizer.
- Knowledge is measured by multiple choice, not open generation.
- RMU may hide knowledge rather than erase it. Our Stage 1 fine-tuning attack brought much of it back; that test is being re-run.
- The published models were trained with the outdated tokenizer. Their model cards say so.

## Links

[Demo](https://huggingface.co/spaces/itssaharsh/relapse) · [Code](https://github.com/itssaharsh/relapse) · [Models](https://huggingface.co/collections/itssaharsh/relapse-unlearned-tiny-aya-models) · [Data](https://huggingface.co/collections/itssaharsh/relapse-runs-logs-and-probe-sets)

## Changelog

- 2026-11-10 to 11: Stage 1 write-up, now in [`docs/archive/`](https://github.com/itssaharsh/relapse/tree/main/docs/archive).
- 2026-11-21: found that Stage 1 used an outdated tokenizer setting. Withdrew the relapse and repair claims, rescored the main models with the official tokenizer, and started a pre-registered experiment.

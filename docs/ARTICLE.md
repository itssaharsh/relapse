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

<!-- stage2:start -->
## Results (Stage 2: pre-registered, official tokenizer)

Every experiment below was registered in this repository before it ran ([1](https://github.com/itssaharsh/relapse/blob/main/results/PREREGISTRATION.md), [2](https://github.com/itssaharsh/relapse/blob/main/results/PREREGISTRATION_2.md), [3](https://github.com/itssaharsh/relapse/blob/main/results/PREREGISTRATION_3.md), [4](https://github.com/itssaharsh/relapse/blob/main/results/PREREGISTRATION_4.md)), and every hypothesis that failed is listed.

**1. Forgetting usually survives the 4-bit file.** In a map of 150 runs across the five Tiny Aya models, 107 of the 109 runs that forgot enough to count kept their forgetting in both Q4_0 and NF4.

**2. But one file can relapse while its siblings hold.** On three models, one 4-bit format brought back at least 5 points of hazardous knowledge while the other formats of the same weights did not:

| Run | fp16 | Q4_0 round trip | NF4 | Real Q4_0 file | Real Q4_K_M file |
|---|---|---|---|---|---|
| Tiny Aya Base, layers 10 to 12, seed 12 | 46.8% | 52.9% (+6.0) **relapse** | 43.8% (-3.1) | 52.9% (+6.0) | 45.6% (-1.2) |
| Tiny Aya Base, layers 10 to 12, seed 43 | 41.6% | 46.8% (+5.3) **relapse** | 37.9% (-3.6) | 46.2% (+4.5) | 39.6% (-2.1) |
| Tiny Aya Fire, layers 13 to 15, seed 44 | 34.8% | 33.2% (-1.6) | 40.1% (+5.3) **relapse** | 33.1% (-1.8) | 36.6% (+1.7) |
| Tiny Aya Water, layers 10 to 12, seed 42 | 38.6% | 42.3% (+3.7) | 46.3% (+7.7) **relapse** | 41.2% (+2.5) | 37.2% (-1.4) |

- The Base relapse reproduced when the seed-12 runs were retrained, and happened again on 1 of 8 fresh seeds.
- Loaded in 4-bit through bitsandbytes, the way NF4 is really used, Fire and Water gave the same rises as above.
- No real Q4_K_M file relapsed. In every case at least one 4-bit format held, so scoring each file before release catches it.
- It is rare and modest, and on Base part of it is the format, not the unlearning: the original Tiny Aya Base also gains about 1.8 points in Q4_0. Net of that, no Base run reaches 5 points, and the typical Base run moves no more than the original does. Fire and Water are the cases that survive this check (their originals fall slightly in NF4).
- The 5-point line is a choice. Two of the three real Base files rose 4.5 and 4.6 points, just under it.

**3. The rise comes from weights the edit never touched.** Rounding only the three edited matrices adds at most 1.7 points in any relapsing run. On Water and Fire the rise comes from the layers in front of the edit (+7.6 of +7.7 points on Water). On Base seed 12 it comes from the layers behind it.

**4. Training on the file closes the gap, at a cost.** In the 5 relapsing runs, the retrained model no longer relapsed in 5, was significantly safer in the file in 4, and stayed within 3 MMLU points in 3. But plain RMU trained for the same extra time did almost as well (4, 4 and 3 of 5), so we cannot separate the repair from longer training, and too much of either forgets far more than intended.

**5. A different tokenizer setting is the larger risk.** Of 45 runs unlearned through the outdated tokenizer setting, 33 scored higher through the official one and 12 lower; 29 rose by at least 5 points. The average rise was +13.3 points at SafeTune's default layers 5 to 7 and +14.1 at layers 10 to 12, but -0.6 at layers 7 to 9. Every Tiny Aya Base run trained this way scored 24.9% (chance) in its own pipeline and 31.2% to 60.9% through the official tokenizer. The direction matters: runs unlearned through the official tokenizer and scored through the outdated one rose by only +1.4 points on average (6 of 47 relapsed).

**6. Where to edit.** Layers 3 to 5 beat the default layers in 9 of 10 comparisons. CircuitKIT ranks that window first on three models and first when its attribution is pooled over the five; its pick for each model alone was unstable and that hypothesis failed. On two further seeds, layers 3 to 5 forgot 5.6 to 13.3 points more on four of the five models (8 of 10 comparisons, p = 0.055, just short of our own line). The comparison holds general knowledge fixed; layers 3 to 5 cost more benign biology than the default layers do.

### The released models

Layers 3 to 5, two seeds per model (the table shows their mean). WMDP-bio is the hazard set (lower is better, 25% is guessing); the other columns should stay high.

| Model | WMDP-bio: original → unlearned | In the real Q4_K_M file | MMLU | Benign biology | With five worked examples | Through the outdated tokenizer | After a fine-tuning attack |
|---|---|---|---|---|---|---|---|
| Tiny Aya Global | 63.4% → 42.6% | 42.1% | 55.4% → 54.1% | 62.5% → 55.1% | 24.7% | 42.5% | 43.2% |
| Tiny Aya Fire | 63.5% → 41.7% | 39.9% | 57.9% → 56.2% | 64.3% → 58.6% | 25.5% | 40.4% | 45.4% |
| Tiny Aya Earth | 62.8% → 43.8% | 43.0% | 56.5% → 54.3% | 62.4% → 56.5% | 26.0% | 44.5% | 46.5% |
| Tiny Aya Water | 62.8% → 44.1% | 43.9% | 57.1% → 55.4% | 63.5% → 57.5% | 25.8% | 42.9% | 48.3% |
| Tiny Aya Base | 60.1% → 36.2% | 36.1% | 57.6% → 56.6% | 65.4% → 58.8% | 25.5% | 37.0% | 56.5% |

- **Holds:** prompt attacks do not bring the knowledge back, reworded and translated questions (Hindi, Swahili, Arabic, Spanish) stay significantly below the original in 40 of 50 cases, and neither the real files nor the outdated tokenizer changes the score by more than 3.3 points.
- **Costs:** benign biology falls by 4.7 to 7.9 points, so the forgetting is not confined to hazardous questions.
- **Breaks:** 25 fine-tuning steps on 64 hazard questions raise the score by at least 5 points on 5 of 10 models and restore Tiny Aya Base almost fully. RMU hides knowledge more than it removes it.

### Every registered hypothesis

| Experiment | Hypothesis | Outcome |
|---|---|---|
| Map | A relapse tends to appear in one format, not both | Held, on 2 relapses |
| Map | A single-format audit misses relapses in both directions | **Failed** in the map (Q4_0 only). The NF4 direction appeared in the follow-up |
| Map | Layers 7 to 9 cost less at equal forgetting | Could not be evaluated as registered |
| Map | Layers 10 to 12 relapse more than layers 7 to 9 | **Failed** (p = 0.077) |
| Relapse follow-up | The Base relapse happens again on fresh seeds | Held: 1 of 8 seeds |
| Relapse follow-up | On Base, Q4_0 moves the score up more than NF4 | Held: 7 of 8 seeds, p = 0.035 |
| Relapse follow-up | A relapse in NF4 and not Q4_0 exists | Held: Fire (1 of 8 seeds), and Water |
| Relapse follow-up | The relapse shows in the real Q4_0 file | Held only for the 1 Base file we had already seen; the two built afterwards rose 4.5 and 4.6 points, under the 5-point line |
| Relapse follow-up | On Base it comes from the layers in front of the edit | **Failed.** On Base it is the layers behind; on Fire and Water it is the layers in front |
| Relapse follow-up | Training on the file repairs it | Gap closed in 5 of 5; file significantly safer in 4; within 3 MMLU points in 3 |
| Relapse follow-up | A rule for how long to train the repair | **Failed** on its test case (Water) |
| Tokenizer | Unlearning through one tokenizer relapses through the other | Held: 33 up, 12 down, p = 0.0012; held-out models 10 to 2, p = 0.019 |
| Tokenizer | The rise is smaller at layers 7 to 9 than at layers 5 to 7 | Held on all five models (16 of 16 pairs). **Not significant on the held-out models alone** (4 of 4, p = 0.062) |
| Tokenizer | Training on both tokenizers narrows the gap | Held on all five (31 pairs against 13, p = 0.005). **Not significant on the held-out models alone** (p = 0.19) |
| Tokenizer | Training on both protects the deployed model | **Failed** (18 pairs against 26) |
| Placement | Layers 3 to 5 beat the default layers 5 to 7 | Held: 9 of 10, p = 0.011 |
| Placement | Layers 7 to 9 beat the default | **Failed** (7 of 10) |
| Placement | Each model's own CircuitKIT window beats the default | **Failed** (6 of 10) |
| Placement | Layers 3 to 5 beat the default again on the final seeds | **Failed narrowly**: 8 of 10, p = 0.055 |

<!-- stage2:end -->

## Method

**Unlearning.** [SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s RMU edits the MLP output projection of three consecutive layers. On hazardous text it pushes activations toward a fixed random vector; on ordinary text (Wikitext) it holds them to a frozen copy of the model. [CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT)'s EAP-IG attribution, pooled over the five models, ranks layers 3 to 5 first; SafeTune's default is 5 to 7.

**Scoring.** Each question is scored from the next-token probabilities of A to D, so no judge model is involved. Runs are compared on the same questions with a paired bootstrap; [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) does this for every real file. Reworded and translated questions pass [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT)'s quality gates.

**Formats.** Real files are built with llama.cpp and read back for scoring. The Q4_0 and NF4 round trips are exact copies of what those formats store for the decoder's linear weights, with the embedding table left at fp16.

## Limits

- Knowledge is measured by multiple choice, not open generation.
- Relapses are rare, so each rests on one or two seeds per model.
- RMU hides knowledge more than it removes it: a short fine-tune on the topic brings much of it back.
- The forgetting also costs 5 to 8 points on benign biology.
- One model family and one unlearning method.

## Links

[Demo](https://huggingface.co/spaces/itssaharsh/relapse) · [Code](https://github.com/itssaharsh/relapse) · [Models](https://huggingface.co/collections/itssaharsh/relapse-unlearned-tiny-aya-models) · [Data](https://huggingface.co/collections/itssaharsh/relapse-runs-logs-and-probe-sets)

## Changelog

- 2026-11-10 to 11: Stage 1 write-up, now in [`docs/archive/`](https://github.com/itssaharsh/relapse/tree/main/docs/archive).
- 2026-11-21: found that Stage 1 used an outdated tokenizer setting. Withdrew the relapse and repair claims, rescored the main models with the official tokenizer, and started a pre-registered experiment.
- 2026-11-22: four pre-registered experiments finished (the relapse map and its follow-up, the tokenizer experiment, placement and the final models). Results above, including eight hypotheses that failed.

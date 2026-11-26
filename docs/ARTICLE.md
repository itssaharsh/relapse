# Relapse: an outdated tokenizer setting overstated unlearning at the default layers, not at layers 3 to 5

Small models like Cohere Labs' [Tiny Aya](https://huggingface.co/CohereLabs/tiny-aya-global) run offline, on phones and laptops, where no server can filter what they say. Any safety has to live in the model's weights, and it has to survive the way the model is deployed: the compressed file people download, and the tokenizer setting their copy loads. We used four Lexsi Labs tools to make Tiny Aya forget hazardous biology and checked both. The paper is [An Outdated Tokenizer Setting Overstated Unlearning Applied at Layers 5–7 in 16 of 16 Runs, but Not at Layers 3–5](https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf).

<!-- brief:start -->
## In brief

An outdated tokenizer setting made the unlearning look deeper than it was. Runs unlearned and audited through it at SafeTune's default layers gave back about half of the audited forgetting (47%) once loaded through the official tokenizer, in 16 of 16 runs that count, and every one of them had also lost more than 3 points of general knowledge where it was audited. At layers 3 to 5 the same change did nothing on the four post-trained variants (0 of 16). The layers are not shown to be the cause: each window had its own learning rates.

| Layers edited (Tiny Aya, RMU with SafeTune) | 5 to 7 (default) | 3 to 5 (released) |
|---|---|---|
| Hazard score as audited, through the outdated tokenizer setting | 39.4% | 35.6% |
| Hazard score as deployed, through the official tokenizer | 50.8% | **36.8%** |
| Runs that relapse, among runs that count | 16 of 16 | **0 of 16** |
| Runs that rise 5 points or more, among all runs | 20 of 20 | 4 of 20 |
| General knowledge lost as audited, then as deployed (MMLU points) | 5.8, then 1.7 | 0.5, then 1.1 |
| General knowledge lost (MMLU points, mean) | 2.3 | 1.6 |
| Benign biology lost (points, mean) | 4.5 | 6.3 |
| Hazard points forgotten per point of benign biology lost | 3.26 | 3.30 |

The hazard score is accuracy on WMDP-bio (lower is better; the original models score 60% to 64%; 25% is guessing). A run counts if it forgot enough without falling to guessing (30% to 55% where it was unlearned); a relapse is a rise of 5 points or more. The top five rows are runs unlearned through the outdated setting and then run through the official one; the rest are the final runs on all five models, through the official tokenizer.

- **The two settings are two public copies of the same model.** Cohere fixed the setting on Hugging Face on September 30, 2026. Kaggle's copy and 418 derived Hugging Face repositories (64,619 downloads) still carried the old one on 2026-11-26; none of those is named as an unlearning model, and we saw the relapse only in our own runs. `python scripts/check_tokenizer.py <repo, folder or GGUF file>`, added with the paper, tells you which setting a copy carries and what to do about it, reading only its tokenizer file or GGUF header.
- **It needs the line breaks.** Scored again with every prompt on a single line (a sixth registered test, scoring only), 0 of 16 default-layer runs relapse and the two settings agree to within 1.6 points on every run.
- **What came back was not only the hazard.** The default-layer runs that relapsed had also lost general knowledge where they were audited (5.8 MMLU points there, 1.7 once deployed). In both tokenizer experiments, runs that had lost 3 or more MMLU points as audited rose by 5 points or more in 72 of 72 cases and the others in 0 of 48, so the layers are confounded with that damage. General knowledge lost in the audit is a warning sign, not an exact test.
- **Tiny Aya Base is the exception.** Through the outdated setting it unlearns down to guessing, which our rule does not count, and it rises through the official tokenizer at every window, by 8.5 to 11.6 points at layers 3 to 5. Counting it, 4 of 20 runs at layers 3 to 5 rise, all four Base.
- **In 4-bit files** the forgetting usually holds (107 of 109 runs in the map). Tiny Aya Base regained 5 to 6 points in Q4_0 on two seeds, and one exploratory run each on Fire and Water regained 5 to 8 points in NF4, while the other formats of the same weights held. Each file has to be scored.
- **What it costs:** benign biology falls too, and placement does not change how much per hazard point forgotten. **What breaks it:** 25 fine-tuning steps on the topic raise the score by 5 points or more on 5 of 10 runs.
- **What we got wrong:** our first-round numbers came from the outdated tokenizer setting. Correcting them is how we found the effect. Every new test was registered before it ran: of 25 hypotheses, 11 held, 2 held in part, 11 failed and 1 could not be tested.
<!-- brief:end -->

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

Every experiment below was registered in this repository before it ran ([1](https://github.com/itssaharsh/relapse/blob/main/results/PREREGISTRATION.md), [2](https://github.com/itssaharsh/relapse/blob/main/results/PREREGISTRATION_2.md), [3](https://github.com/itssaharsh/relapse/blob/main/results/PREREGISTRATION_3.md), [4](https://github.com/itssaharsh/relapse/blob/main/results/PREREGISTRATION_4.md), [5](https://github.com/itssaharsh/relapse/blob/main/results/PREREGISTRATION_5.md), [6](https://github.com/itssaharsh/relapse/blob/main/results/PREREGISTRATION_6.md)), and every hypothesis that failed is listed. Every Stage 2 job downloaded the main branch of `CohereLabs/tiny-aya-<variant>` when it started; by the job start times and the repositories' history that was the revision carrying Cohere's tokenizer fix (Global `af89d219b5`, Fire `73500c80fc`, Earth `0624fafbe2`, Water `41a99216d1`, Base `d62ac17e24`, all committed September 30, 2026), as the paper's appendix sets out.

**1. Forgetting usually survives the 4-bit file.** In a map of 150 runs across the five Tiny Aya models, 107 of the 109 runs that forgot enough to count kept their forgetting in both Q4_0 and NF4.

**2. But one file can relapse while its siblings hold.** On three models, one 4-bit format brought back at least 5 points of hazardous knowledge while the other formats of the same weights did not:

| Run | fp16 | Q4_0 round trip | NF4 | Real Q4_0 file | Real Q4_K_M file |
|---|---|---|---|---|---|
| Tiny Aya Base, layers 10 to 12, seed 12 | 46.8% | 52.9% (+6.0) **relapse** | 43.8% (-3.1) | 52.9% (+6.0) | 45.6% (-1.2) |
| Tiny Aya Base, layers 10 to 12, seed 43 | 41.6% | 46.8% (+5.3) **relapse** | 37.9% (-3.6) | 46.2% (+4.5) | 39.6% (-2.1) |
| Tiny Aya Fire, layers 13 to 15, seed 44 | 34.8% | 33.2% (-1.6) | 40.1% (+5.3) **relapse** | 33.1% (-1.8) | 36.6% (+1.7) |
| Tiny Aya Water, layers 10 to 12, seed 42 | 38.6% | 42.3% (+3.7) | 46.3% (+7.7) **relapse** | 41.2% (+2.5) | 37.2% (-1.4) |

- The Base relapse reproduced when the seed-12 runs were retrained, and happened again on 1 of 8 fresh seeds. The Fire and Water cases are one run each, never retrained, and we treat them as exploratory; by an exact McNemar test with a Holm correction over the cells screened with them they are not noise ([`scripts/mcnemar_holm.py`](scripts/mcnemar_holm.py)).
- Loaded in 4-bit through bitsandbytes, the way NF4 is really used, Fire and Water gave the same rises as above.
- Scored inside llama.cpp's own runtime on 600 questions (an exploratory check on the first Base case), the answers matched our scoring on at least 99.8% of questions at full precision. On those questions Q4_0 raised the original model by +3.5 points and the unlearned one by +3.2.
- No real Q4_K_M file relapsed. In every case at least one 4-bit format held, so scoring each file before release catches it.
- It is rare and modest, and on Base part of it is the format, not the unlearning: the original Tiny Aya Base also gains about 1.8 points in Q4_0. Net of that, no Base run reaches 5 points, and the typical Base run moves no more than the original does. Fire and Water are the cases that survive this check (their originals fall slightly in NF4).
- The 5-point line is a choice. Two of the three real Base files rose 4.5 and 4.6 points, just under it.

**3. The rise comes from weights the edit never touched.** Rounding only the three edited matrices adds at most 1.7 points in any relapsing run. On Water and Fire the rise comes from the layers in front of the edit (+7.6 of +7.7 points on Water). On Base seed 12 it comes from the layers behind it.

**4. Training on the file closes the gap, at a cost.** In the 5 relapsing runs, the retrained model no longer relapsed in 5, was significantly safer in the file in 4, and stayed within 3 MMLU points in 3. But plain RMU trained for the same extra time did almost as well (4, 4 and 3 of 5), so we cannot separate the repair from longer training, and too much of either forgets far more than intended.

**5. A different tokenizer setting is the larger risk.** Of 45 runs unlearned through the outdated tokenizer setting, 33 scored higher through the official one and 12 lower; 29 rose by at least 5 points. The average rise was +13.3 points at SafeTune's default layers 5 to 7 and +14.1 at layers 10 to 12, but -0.6 at layers 7 to 9. Every Tiny Aya Base run trained this way scored 24.9% (chance) in its own pipeline and 31.2% to 60.9% through the official tokenizer. The direction matters: runs unlearned through the official tokenizer and scored through the outdated one rose by only +1.4 points on average (6 of 47 relapsed).

**Default layers against layers 3 to 5, at the learning rates tested.** In a further registered test on new seeds, runs unlearned through the outdated tokenizer at SafeTune's default layers relapsed in 16 of 16 cases (+11.4 points on average). At layers 3 to 5, the window we release, 0 of 16 relapsed (+1.2 on average), and at layers 7 to 9, 0 of 16. Scored through the official tokenizer, the layers 3 to 5 runs sit at 36.8% and the default-layer runs at 50.8%. That is on the four post-trained variants. Tiny Aya Base is the exception, and our rule for which runs count removes it: at every window it drops to guessing level in the outdated pipeline and comes back up through the official one, at layers 3 to 5 by 8.5 to 11.6 points. Counting every run, 20 of 20 rose by 5 points or more at the default layers (mean rise +12.1), 4 of 20 at layers 3 to 5 (+3.0) and 4 of 20 at layers 7 to 9 (+1.9); outside the default layers all of them are Base. The effect was measured on multiple-choice prompts, whose line breaks and digit groups are where the two settings split text differently.

**What came back was not only the hazard** (an exploratory reading of the same runs). Where they were audited, through the outdated tokenizer, the default-layer runs had also lost 5.8 MMLU points, none of them within our 3-point line; through the official tokenizer they lost 1.7. At layers 3 to 5 the loss was 0.5 and 1.1. Tiny Aya Base scored 23.1% to 24.0% on MMLU as audited: the edit had wrecked it there, not unlearned it. Over both tokenizer experiments, runs trained through the outdated setting that had lost 3 or more MMLU points as audited rose by 5 points or more in 72 of 72 cases, and the others in 0 of 48. About half of the audited forgetting came back (47%; 35% to 57% per run), and the deployed runs land where runs unlearned through the official tokenizer at the same layers, seeds and learning rates land (50.9% against 52.1%): the outdated audit overstated the forgetting, it did not lose real forgetting at load. By model and seed the count is 8 of 8 at the default layers against 0 of 8 at layers 3 to 5. So the layers are confounded with how much the edit damaged the model in the pipeline it was audited in, and the pattern is not ordered by depth: layers 10 to 12 relapsed like the default layers, layers 7 to 9 held. General knowledge lost in the audit is a warning sign, not a test: the rule is fitted on these runs, and our Stage 1 model at layers 10 to 12 had lost only 1.5 MMLU points where it was audited and still rose 10.6.

**What the two settings do to the text** (counted without any model, [`scripts/token_differences.py`](scripts/token_differences.py)). They give different tokens for all 1,273 WMDP-bio prompts and all 1,000 MMLU prompts, and for 496 of the WMDP-bio questions without the multiple-choice layout. The outdated setting splits every digit apart, separates punctuation from the word it joins, and loses the pieces that join punctuation to a line break.

**It needs the line breaks.** The effect above was measured on prompts with the question and each option on its own line. In a sixth registered test we scored the same 60 saved runs again with every line break removed, through both settings (nothing was trained). The default-layer runs then score 46.6% through the outdated setting and 47.4% through the official one: a rise of +0.8 points against +11.4 in the multiple-choice layout, and 0 of 16 relapse (we had expected at least half to; that hypothesis failed). No run had lost 3 MMLU points through the outdated setting, and Tiny Aya Base is not at guessing level there. So the overstated audit needs both the outdated setting and the line breaks of the multiple-choice layout; why is not explained.

**The outdated setting is still widespread.** The two settings are two public copies of the same model: Cohere changed `tokenizer_class` on Hugging Face on September 30, 2026, and copies made before that keep the old value. On 2026-11-26 we read the tokenizer file, or the GGUF header, of every public copy we could find, without downloading weights ([`scripts/check_tokenizer.py`](scripts/check_tokenizer.py), [`results/survey/`](https://github.com/itssaharsh/relapse/blob/main/results/survey/)): 1,275 Hugging Face repositories tied to a Tiny Aya model, and Kaggle Models. Cohere's five repositories carry the official setting. All 5 Kaggle variations carry the outdated one. Of 915 derived Hugging Face copies whose setting could be read, 423 carry it: 5 were our own Stage 1 models, which we corrected on November 27, 2026, and the other 418, mostly fine-tunes and adapters, have 64,619 downloads (7,133 in the last 30 days). Loaded with `transformers` 5.20.2, 12 of 12 sampled copies split a test prompt differently from the official tokenizer. GGUF files are nearly all safe (2 of 77 name another pre-tokenizer). This counts exposure, not harm: we saw the relapse only in our own runs, 390 of those copies were last modified before the fix and 28 after it, none is named as an unlearning model and 6 are named as safety-tuned, and which direction of mismatch a copy faces depends on the `transformers` version it was trained with, which its files do not record. Who is exposed is anyone who unlearns or audits through a copy with the outdated setting and ships a model that users load through the official one.

**6. Where to edit.** Two registered tests compared layers 3 to 5 with SafeTune's default layers; one passed and one narrowly missed. In the first, layers 3 to 5 forgot more within 3 MMLU points in 9 of 10 comparisons (p = 0.011). In the second, on fresh seeds, they forgot 5.6 to 13.3 points more on four of the five models (8 of 10 comparisons, p = 0.055, short of our own line). What placement buys is more forgetting per MMLU point (20.8 hazard points for 1.6 MMLU points, against 14.6 for 2.3). Layers 3 to 5 also did not relapse through the tokenizer where the default layers did, but the default-layer runs were the ones whose audit showed MMLU damage, so the layer is not shown to be the reason. Placement does not buy selectivity within biology: benign biology falls by 6.3 points against 4.5, the same 3.30 and 3.26 hazard points forgotten per point of benign biology lost. CircuitKIT ranks layers 3 to 5 first on three models and first when its attribution is pooled over the five; its pick for each model alone was unstable and that hypothesis failed (with 160 prompts instead of 64, an exploratory repeat, the per-model pick changed on three of five models while the pooled pick stayed layers 3 to 5).

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

Of 25 registered hypotheses, 11 held, 2 held in part, 11 failed and 1 could not be tested. Where a pre-registration names the held-out models as the confirmatory test, a hypothesis that holds on all five models and not on the held-out ones counts as failed.

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
| Placement and tokenizer | Unlearned through the outdated tokenizer, the rise is smaller at layers 3 to 5 than at the default layers | Held: 16 of 16 pairs |
| Placement and tokenizer | Fewer than 1 in 5 runs at layers 3 to 5 relapse through the tokenizer | Held: 0 of 16 |
| Placement and tokenizer | Layers 7 to 9 again do not relapse and the default layers again do | Held: 0 of 16 and 16 of 16 |
| Placement | Layers 3 to 5 beat the default layers 5 to 7 | Held: 9 of 10, p = 0.011 |
| Placement | Layers 7 to 9 beat the default | **Failed** (7 of 10) |
| Placement | Each model's own CircuitKIT window beats the default | **Failed** (6 of 10) |
| Placement | Layers 3 to 5 beat the default again on the final seeds | **Failed narrowly**: 8 of 10, p = 0.055 |
| Single-line prompts | With single-line prompts the default-layer rise is smaller than in the multiple-choice layout | Held: 16 of 16 runs |
| Single-line prompts | At least half of the default-layer runs still relapse with single-line prompts | **Failed**: 0 of 16 |
| Single-line prompts | Fewer than 1 in 5 runs at layers 3 to 5 relapse with single-line prompts | Held: 0 of 16 |

<!-- stage2:end -->

## Method

**Unlearning.** [SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s RMU edits the MLP output projection of three consecutive layers. On hazardous text it pushes activations toward a fixed random vector; on ordinary text (Wikitext) it holds them to a frozen copy of the model. [CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT)'s EAP-IG attribution, pooled over the five models, ranks layers 3 to 5 first; SafeTune's default is 5 to 7.

**Scoring.** Each question is scored from the next-token probabilities of A to D, so no judge model is involved. Runs are compared on the same questions with a paired bootstrap; [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) does this for every real file. Reworded and translated questions pass [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT)'s quality gates.

**Formats.** Real files are built with llama.cpp and read back for scoring. The Q4_0 and NF4 round trips are exact copies of what those formats store for the decoder's linear weights, with the embedding table left at fp16.

## Limits

- Knowledge is measured by multiple choice, not open generation. The tokenizer effect belongs to the multiple-choice layout: with single-line prompts it is absent, and we have no mechanism for it.
- Layers 3 to 5 did not relapse through the tokenizer on the four post-trained variants only (Tiny Aya Base rose at every window), and the default-layer runs that did relapse were the ones whose audit showed MMLU damage, so the layer is not shown to be the cause.
- Relapses in 4-bit files are rare, so each rests on one or two seeds per model; the Fire and Water cases are one run each.
- Placement buys more forgetting per MMLU point, not per point of benign biology.
- RMU hides knowledge more than it removes it: a short fine-tune on the topic brings much of it back.
- The forgetting also costs benign biology (see the tables above).
- One model family and one unlearning method.

## Links

[Paper](https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf) · [Tokenizer checker](https://github.com/itssaharsh/relapse/blob/main/scripts/check_tokenizer.py) · [Demo](https://huggingface.co/spaces/itssaharsh/relapse) · [Code](https://github.com/itssaharsh/relapse) · [Models](https://huggingface.co/collections/itssaharsh/relapse-unlearned-tiny-aya-models) · [Data](https://huggingface.co/collections/itssaharsh/relapse-runs-logs-and-probe-sets)

## Changelog

- 2026-11-10 to 11: Stage 1 write-up, now in [`docs/archive/`](https://github.com/itssaharsh/relapse/tree/main/docs/archive).
- 2026-11-21: found that Stage 1 used an outdated tokenizer setting. Withdrew the relapse and repair claims, rescored the main models with the official tokenizer, and started a pre-registered experiment.
- 2026-11-22: four pre-registered experiments finished (the relapse map and its follow-up, the tokenizer experiment, placement and the final models). Results above, with every hypothesis that failed.
- 2026-11-22: a fifth pre-registered experiment (placement against the tokenizer).
- 2026-11-26: the paper leads with the tokenizer-by-layer result; the count over every run, with Tiny Aya Base, is stated beside the filtered one; a survey of public copies and a tokenizer checker are added.
- 2026-11-26: a sixth pre-registered test, scoring only: with single-line prompts the tokenizer effect is absent.

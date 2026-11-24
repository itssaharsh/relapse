# Relapse: forget the hazard, keep the rest, even in the 4-bit file

Relapse makes Cohere Labs' Tiny Aya forget hazardous biology with Lexsi Labs' open-source safety stack, then checks the files people actually download to see whether it is still forgotten and what that cost.

**Paper:** [paper/main.pdf](paper/main.pdf) · **Live demo:** [huggingface.co/spaces/itssaharsh/relapse](https://huggingface.co/spaces/itssaharsh/relapse) · **Write-up:** [on Hugging Face](https://huggingface.co/spaces/itssaharsh/relapse-article) · **Models:** [collection](https://huggingface.co/collections/itssaharsh/relapse-unlearned-tiny-aya-models) · **Data and logs:** [collection](https://huggingface.co/collections/itssaharsh/relapse-runs-logs-and-probe-sets)

<!-- brief:start -->
## In brief

SafeTune's default recipe looks unlearned where it was built and comes back when deployed. Editing at layers 3 to 5, the window CircuitKIT ranks first across the model family, forgets more and does not come back.

| Tiny Aya, RMU with SafeTune | Layers 5 to 7 (default) | Layers 3 to 5 (released) |
|---|---|---|
| Hazard score in the pipeline it was unlearned in | 39.4% | 35.6% |
| Hazard score once deployed through the official tokenizer | 50.8% | **36.8%** |
| Runs that relapse through the tokenizer | 16 of 16 | **0 of 16** |
| Hazard score, official tokenizer throughout | 47.8% | **41.7%** |
| Relapses in Q4_0 or NF4 (of 20 checks) | 0 | 0 |
| General knowledge lost (MMLU points, mean) | 2.3 | 1.6 |
| Benign biology lost (points, median) | 3.4 | 6.0 |

The hazard score is accuracy on WMDP-bio (lower is better; the original models score 60% to 64%; 25% is guessing). The top three rows are runs unlearned through an outdated tokenizer setting and then run through the official one (16 runs per column, four models). The rest are the final runs on all five models.

- **In 4-bit files** the forgetting usually holds (107 of 109 runs in the map). On three models one format regained 5 to 8 points while the others held, so each file has to be scored.
- **What it costs:** benign biology falls too, more at layers 3 to 5. **What breaks it:** a short fine-tune on the topic.
- **What we got wrong:** our first-round numbers came from an outdated tokenizer setting. We corrected them in public, registered every new test before running it, and report the ten hypotheses that failed.
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

Every experiment below was registered in this repository before it ran ([1](results/PREREGISTRATION.md), [2](results/PREREGISTRATION_2.md), [3](results/PREREGISTRATION_3.md), [4](results/PREREGISTRATION_4.md), [5](results/PREREGISTRATION_5.md)), and every hypothesis that failed is listed.

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
- Scored inside llama.cpp's own runtime on 600 questions (an exploratory check on the first Base case), the answers matched our scoring on at least 99.8% of questions at full precision. On those questions Q4_0 raised the original model by +3.5 points and the unlearned one by +3.2.
- No real Q4_K_M file relapsed. In every case at least one 4-bit format held, so scoring each file before release catches it.
- It is rare and modest, and on Base part of it is the format, not the unlearning: the original Tiny Aya Base also gains about 1.8 points in Q4_0. Net of that, no Base run reaches 5 points, and the typical Base run moves no more than the original does. Fire and Water are the cases that survive this check (their originals fall slightly in NF4).
- The 5-point line is a choice. Two of the three real Base files rose 4.5 and 4.6 points, just under it.

**3. The rise comes from weights the edit never touched.** Rounding only the three edited matrices adds at most 1.7 points in any relapsing run. On Water and Fire the rise comes from the layers in front of the edit (+7.6 of +7.7 points on Water). On Base seed 12 it comes from the layers behind it.

**4. Training on the file closes the gap, at a cost.** In the 5 relapsing runs, the retrained model no longer relapsed in 5, was significantly safer in the file in 4, and stayed within 3 MMLU points in 3. But plain RMU trained for the same extra time did almost as well (4, 4 and 3 of 5), so we cannot separate the repair from longer training, and too much of either forgets far more than intended.

**5. A different tokenizer setting is the larger risk.** Of 45 runs unlearned through the outdated tokenizer setting, 33 scored higher through the official one and 12 lower; 29 rose by at least 5 points. The average rise was +13.3 points at SafeTune's default layers 5 to 7 and +14.1 at layers 10 to 12, but -0.6 at layers 7 to 9. Every Tiny Aya Base run trained this way scored 24.9% (chance) in its own pipeline and 31.2% to 60.9% through the official tokenizer. The direction matters: runs unlearned through the official tokenizer and scored through the outdated one rose by only +1.4 points on average (6 of 47 relapsed).

**Where the edit sits decides it.** In a further registered test on new seeds, runs unlearned through the outdated tokenizer at SafeTune's default layers relapsed in 16 of 16 cases (+11.4 points on average). At layers 3 to 5, the window we release, 0 of 16 relapsed (+1.2 on average), and at layers 7 to 9, 0 of 16. Scored through the official tokenizer, the layers 3 to 5 runs sit at 36.8% and the default-layer runs at 50.8%. Tiny Aya Base is the exception: it drops to guessing level in the outdated pipeline at every window and comes back up through the official one.

**6. Where to edit.** Layers 3 to 5 beat the default layers in 9 of 10 comparisons. CircuitKIT ranks that window first on three models and first when its attribution is pooled over the five; its pick for each model alone was unstable and that hypothesis failed (with 160 prompts instead of 64, an exploratory repeat, the per-model pick changed on three of five models while the pooled pick stayed layers 3 to 5). On two further seeds, layers 3 to 5 forgot 5.6 to 13.3 points more on four of the five models (8 of 10 comparisons, p = 0.055, just short of our own line). The comparison holds general knowledge fixed; layers 3 to 5 cost more benign biology than the default layers do.

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
| Placement and tokenizer | Unlearned through the outdated tokenizer, the rise is smaller at layers 3 to 5 than at the default layers | Held: 16 of 16 pairs |
| Placement and tokenizer | Fewer than 1 in 5 runs at layers 3 to 5 relapse through the tokenizer | Held: 0 of 16 |
| Placement and tokenizer | Layers 7 to 9 again do not relapse and the default layers again do | Held: 0 of 16 and 16 of 16 |
| Placement | Layers 3 to 5 beat the default layers 5 to 7 | Held: 9 of 10, p = 0.011 |
| Placement | Layers 7 to 9 beat the default | **Failed** (7 of 10) |
| Placement | Each model's own CircuitKIT window beats the default | **Failed** (6 of 10) |
| Placement | Layers 3 to 5 beat the default again on the final seeds | **Failed narrowly**: 8 of 10, p = 0.055 |

<!-- stage2:end -->

## How it works

| Step | Tool | What it does here |
|---|---|---|
| Build probe sets | [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) | Quality gates on reworded and translated WMDP-bio questions, used to check that the forgetting is not tied to one wording or language |
| Find where to edit | [CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT) | EAP-IG attribution across Tiny Aya's 36 layers. Pooled over the five models it ranks layers 3 to 5 first, the window that beat SafeTune's default |
| Remove the knowledge | [SafeTune](https://github.com/Lexsi-Labs/SafeTune) | RMU unlearning on the WMDP bio forget corpus, with a Wikitext retain set |
| Check every file | [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) | Paired bootstrap comparisons of each real llama.cpp file against the same weights at full precision |

Real files are made with llama.cpp's own tools and read back for scoring. The Q4_0 and NF4 round trips are exact copies of what those formats store for the decoder's linear weights, with the embedding table left at fp16. The pre-registered analyses use a paired bootstrap over the same questions.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests               # format round trips, and the numbers on this page against the data
python scripts/analyze_map.py           # each pre-registered analysis reads results/ and prints its hypotheses
python scripts/analyze_shift.py
python scripts/analyze_place.py
python scripts/analyze_final.py
python scripts/build_readme.py          # rebuilds the results block of this page
```

| Folder | What is in it |
|---|---|
| [`results/PREREGISTRATION.md`](results/PREREGISTRATION.md) and `_2`, `_3`, `_4` | The hypotheses, each committed before its runs |
| [`results/map/`](results/map/), [`deep/`](results/deep/), [`shift/`](results/shift/), [`place/`](results/place/), [`final/`](results/final/) | Scores of every run, per question, and the summaries the analyses write |
| [`results/correct_tokenizer/`](results/correct_tokenizer/) | The Stage 1 models rescored, and the tokenizer control |
| [`hf_jobs/`](hf_jobs/), [`kaggle/`](kaggle/) | The jobs and the experiment scripts they run |
| [`paper/`](paper/) | The paper |
| [`space/`](space/) | The demo page |
| [`docs/archive/`](docs/archive/), [`results/raw/`](results/raw/), [`scripts/archive/`](scripts/archive/) | Stage 1, superseded, kept for the record |

## Limits

- Knowledge is measured by multiple choice, not open generation.
- Relapses are rare, so each rests on one or two seeds per model.
- RMU hides knowledge more than it removes it: a short fine-tune on the topic brings much of it back.
- The forgetting also costs 5 to 8 points on benign biology.
- One model family and one unlearning method.
- The models published in Stage 1 were trained with the outdated tokenizer; their cards say so. The Stage 2 models replace them.

Built during the Lexsi AI Safety and Alignment Hackathon (November 2026), with Claude Code as the coding agent.

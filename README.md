# Relapse: forget the hazard, keep the rest, even in the 4-bit file

Relapse makes Cohere Labs' Tiny Aya forget hazardous biology with Lexsi Labs' open-source safety stack, then checks every file people download to see whether it is still forgotten and what that cost.

You want a small model that runs offline on a phone and will not help anyone build a bioweapon. Unlearning can remove the dangerous knowledge, but two things usually go wrong. Nobody runs the checkpoint that was tested; people run a 4-bit copy of it. And unlearning costs some of everything else the model knows. Relapse measures both, and finds a recipe where the 4-bit file keeps the forgetting and gives up nothing measurable for it. It chains four Lexsi Labs repos (CuratorKIT, CircuitKIT, SafeTune, AuditKIT), and every model, dataset and log is on Hugging Face.

**The result:** unlearning at the layers [CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT) pointed to forgot as much as [SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s default layers and kept more of everything else. Tiny Aya's hazardous-biology score fell from 63% to about 37% in every file format. Head to head in the llama.cpp Q4_K_M file, the CircuitKIT-layer model kept 5.6 more points of general knowledge than the default-layer model (p = 0.001), and 3.9 more in a second training seed (p = 0.006). The forgetting held in all 7 formats, in four other languages and against prompt attacks. It did not hold against fine-tuning, which is in the list of what didn't work below.

[![Relapse: the forget report flipping from the default layers to the CircuitKIT layers](docs/demo.gif)](https://huggingface.co/spaces/itssaharsh/relapse)

*35-second walk-through: [docs/demo.mp4](docs/demo.mp4)*

**Live demo:** [huggingface.co/spaces/itssaharsh/relapse](https://huggingface.co/spaces/itssaharsh/relapse) ([full screen](https://itssaharsh-relapse.static.hf.space/index.html)) · **Write-up:** [the article](https://github.com/itssaharsh/relapse/blob/main/docs/ARTICLE.md) · **Models:** [collection](https://huggingface.co/collections/itssaharsh/relapse-unlearned-tiny-aya-models) · **Data and logs:** [collection](https://huggingface.co/collections/itssaharsh/relapse-runs-logs-and-probe-sets)

## Results

**What worked: unlearning at the layers CircuitKIT pointed to.** We removed hazardous biology knowledge from [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B parameters, 70 languages) with SafeTune's RMU, scored on WMDP-bio (1,273 four-choice questions, 25% is guessing) and MMLU (1,000 questions). RMU edits the MLPs of three consecutive early layers, and SafeTune's default is layers 5 to 7. CircuitKIT's attribution on WMDP-bio gave layers 7 to 9 the most MLP attribution of any three-layer window in layers 1 to 12 (0.30, against 0.23 for 5 to 7), so we put the same edit there.

Head to head on the same questions, with AuditKIT's paired bootstrap, the CircuitKIT-layer model kept more general knowledge in all four comparisons, significantly in three:

<!-- recipes:start -->
| Seed | File | WMDP-bio: default / CircuitKIT layers | MMLU: default / CircuitKIT layers | MMLU kept extra (95% interval) | p |
|---|---|---|---|---|---|
| 1 | fp16 | 38.7% / 37.0% | 51.5% / 53.1% | +1.6 (-1.1 to +4.3) | 0.24 |
| 1 | Q4_K_M | 37.2% / 37.2% | 49.3% / 54.9% | +5.6 (+2.8 to +8.3) | 0.001 |
| 2 | fp16 | 44.5% / 45.6% | 51.7% / 55.7% | +4.0 (+1.3 to +6.6) | 0.003 |
| 2 | Q4_K_M | 41.1% / 45.4% | 51.5% / 55.4% | +3.9 (+1.3 to +6.6) | 0.006 |
<!-- recipes:end -->

The two models forgot the same amount in the first three rows (the hazard scores differ by 1.7 points or less, p = 0.33 or more). In the last row the default-layer model forgot 4.3 points more (p = 0.009).

**Against the original model**, in the same file format, the CircuitKIT-layer model cost -0.1 to 2.8 points of general knowledge across the seven formats in the first seed, and -0.6 to 0.7 across five formats in the second. The default-layer model cost 3.7 to 6.4 and 3.0 to 3.7. In the Q4_K_M file the CircuitKIT-layer model's cost was not measurable in either seed (+0.1 points, 95% interval -1.9 to +2.0; +0.6, -1.1 to +2.3). At fp16 in the first seed it was 2.2 points (p = 0.027).

**The sweep.** We ran 8 RMU settings at each set of layers. Among the runs that got WMDP-bio to 40% or below, the default layers cost 3.8 to 19.9 points of general knowledge at fp16, 4.9 to 14.1 in Q4_0 and 5.6 to 16.5 in NF4. The CircuitKIT layers cost 0.9 to 2.7, 2.5 to 3.5 and 1.4 to 3.3. The ranges stay apart for cut-offs from 35% to 42%; at 45% they overlap in Q4_0. The two sets were not tuned identically: the default layers got learning rates from 2e-5 to 3e-4 and one lower retain weight, the CircuitKIT layers got 1e-4 to 2e-4, one higher retain weight and one stronger steering setting, and runs used 250 or 300 batches.

![Every run: general knowledge kept against hazardous biology still answered, in the 4-bit NF4 file](docs/tradeoff.png)

**The forgetting held in every file, language and prompt we tried.** Across all 16 runs and every format, no hazard score rose by more than 2.0 points from its fp16 value (55 run-and-file pairs; the largest fall was 3.5). AuditKIT's paired fp16-to-Q4_K_M tests found no increase (the largest was +2.0 points, p = 0.12). Translated into Hindi, Arabic, Spanish and Swahili or reworded in English, the questions stayed forgotten: the original scores 46% to 57% per language, the CircuitKIT version 27% to 33%. Five worked examples in the prompt lifted the original to 63.5% but pushed the unlearned models down to 25.2% to 27.7%, and an expert persona left them at 32.2% to 35.0%.

<!-- results:start -->
| File (WMDP-bio / MMLU) | Original | Default layers | CircuitKIT layers | CircuitKIT + 4-bit aware | Default, seed 2 | CircuitKIT, seed 2 | Aggressive |
|---|---|---|---|---|---|---|---|
| fp16 | 63.3% / 55.4% | 38.7% / 51.6% | 36.8% / 53.2% | 34.0% / 54.5% | 44.4% / 51.7% | 45.6% / 55.7% | 25.4% / 43.5% |
| bf16 | 63.3% / 55.4% | 38.6% / 51.4% | 36.9% / 53.1% | 34.2% / 54.5% | not run | not run | 25.4% / 43.5% |
| Q8_0 | 63.2% / 55.1% | 38.4% / 51.4% | 37.3% / 53.2% | 34.6% / 54.4% | 44.4% / 52.1% | 45.4% / 55.7% | 26.4% / 43.8% |
| Q6_K | 63.2% / 55.6% | 39.1% / 51.7% | 36.4% / 53.8% | 34.9% / 53.9% | not run | not run | 26.9% / 42.9% |
| Q4_K_M | 63.4% / 54.8% | 37.2% / 49.3% | 37.1% / 54.9% | 35.3% / 54.5% | 41.2% / 51.6% | 45.3% / 55.4% | 27.3% / 40.7% |
| Q4_0 | 61.1% / 55.5% | 36.8% / 50.6% | 35.9% / 52.7% | 35.6% / 52.9% | 42.3% / 52.5% | 43.8% / 54.8% | 26.8% / 41.4% |
| NF4 | 61.1% / 52.7% | 37.2% / 46.3% | 35.3% / 51.2% | 34.7% / 50.8% | 45.2% / 49.7% | 44.1% / 52.4% | 26.7% / 36.2% |
<!-- results:end -->

WMDP-bio accuracy is the knowledge that should be gone (lower is better; 25% is guessing). MMLU accuracy is the general knowledge that should stay (higher is better). Every cell is measured on the same questions; the Space shows the per-file details and the per-language scores.

## What we tried that didn't work

These are the steps that led to the recipe above. All runs are in [`results/raw/`](results/raw/).

- **SafeTune's RMU defaults barely moved Tiny Aya.** With retain weight 1200, learning rate 5e-5 and 150 batches, WMDP-bio went from 63.3% to 62.5%. A retain weight of 100 with 250 to 300 batches is what worked.
- **Pushing harder at the default layers forgot everything and broke the model.** At learning rate 1e-4 the score reached 25.4% (guessing), but MMLU fell to 43.5% and kept falling at 4-bit (36.2% in NF4). That is the "Aggressive" column.
- **SafeTune's NPO was not selective on Tiny Aya.** At learning rate 2e-4 it only reached 58.6%. At 5e-4 it reached 24.7% but took MMLU down to 23.4%, which is guessing too.
- **Training through a 4-bit quantizer did nothing at the default layers** (38.9% against 38.7%, same MMLU). At the CircuitKIT layers it forgot a little more in both pairs we ran (34.0% against 36.8%, and 32.4% against 33.6%). That is well inside the seed-to-seed spread, so we do not claim an effect. It is the "CircuitKIT + 4-bit aware" column.
- **4-bit compression did not bring the knowledge back** for any recipe we ran, so the question became what the 4-bit file pays for the forgetting.
- **The default layers' cost grew at 4-bit in one seed only.** In the first seed it went from 3.8 points at fp16 to 5.5 in Q4_K_M and 6.4 in NF4. In the second it stayed at 3.7, 3.3 and 3.0. What held in both seeds is the gap between the two recipes.
- **The forgetting did not survive fine-tuning.** 25 LoRA steps on 64 WMDP-bio questions with their answers brought the score on the other 1,204 questions back to 52.4% (default layers), 57.1% (CircuitKIT layers) and 55.5% (CircuitKIT, 4-bit aware), against 62.0% for the original. Before fine-tuning they scored 37.4%, 36.5% and 34.1% on those questions, so the CircuitKIT-layer models came back further. This suggests RMU hides the knowledge from prompts and from compression without erasing it, as [Łucki et al.](https://arxiv.org/abs/2409.18025) found on other models. Making it survive fine-tuning is the tamper-resistance problem, and the next thing to try.
- **How much a run forgets depends on the seed.** The same settings reached 36.8% in one seed and 45.6% in the other, which is why the results report both.

## For judges: one minute

1. Open the [Space](https://huggingface.co/spaces/itssaharsh/relapse). It opens on the CircuitKIT-layer model in the llama.cpp Q4_K_M file: still forgotten, nothing else lost, with the default layers' cost in the same file beside it.
2. Click **Default layers, same file**: the same forgetting now costs 5.5 points of general knowledge, and the page turns from green to red. **Ask in Swahili** shows another language.
3. Scroll once: the chart of all 16 runs, then the prompt and fine-tuning attacks.

Every number on the page comes from [`space/data/results.json`](space/data/results.json), which [`scripts/build_results.py`](scripts/build_results.py) rebuilds from the raw runs.

## How it works

Four Lexsi Labs repositories in one pipeline, on a Cohere Labs model, with every artifact on Hugging Face:

| Step | Tool | What it did here |
|---|---|---|
| Build the probe sets | [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) | Schema, dedup and cleaning gates (with provenance manifests) on 707 WMDP-bio questions translated into Hindi, Arabic, Spanish and Swahili or reworded in English by Tiny Aya Global, kept only when the back-translation scored chrF 35 or more against the source |
| Find where the knowledge flows | [CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT) | EAP-IG attribution on WMDP-bio over Tiny Aya's 36 layers: MLPs 7 and 9 carry the answers, RMU's default layers 5 and 6 carry little, so the edit moved to layers 7 to 9 |
| Remove it | [SafeTune](https://github.com/Lexsi-Labs/SafeTune) | RMU on the WMDP bio forget corpus with a wikitext retain set, 8 settings at each set of layers; NPO was tried too and was not selective on Tiny Aya |
| Check every file | [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) | Scores each model in each format by answer log-likelihood and runs the paired bootstrap comparisons (fp16 against Q4_K_M, unlearned against original) |

The formats are made the way people make them: the bf16 checkpoint, llama.cpp's `convert_hf_to_gguf.py` and `llama-quantize` for Q6_K and Q4_K_M (read back into PyTorch for scoring), bit-exact re-implementations of llama.cpp's Q8_0 and Q4_0 and of bitsandbytes' NF4 for the rest. The quantization code is in [`kaggle/common.py`](kaggle/common.py), the helpers every run used, and [`tests/`](tests/) checks it bit for bit against llama.cpp's own reference code.

## Model, data and metrics, and why these

- **Tiny Aya Global** is the hackathon's model and the one Cohere ships as GGUF files for phones, so its 4-bit files are what people will run. Global is the balanced variant across its 70 languages.
- **Hazardous biology** is the standard unlearning target for safety: the WMDP benchmark and its bio forget corpus were built for it, and SafeTune's RMU comes from the same work. Wikitext is RMU's usual retain set.

- **WMDP-bio accuracy** is the standard measure of hazardous biology knowledge in the unlearning literature (the RMU paper uses it). It is multiple choice, so it needs no judge model and gives the same answer every run.
- **MMLU accuracy** on a fixed sample of 1,000 questions is the standard measure of what unlearning costs.
- **Probability on the right answer** for both, a smoother signal than accuracy, is in the raw runs.
- **The same questions in four languages and reworded** test whether the forgetting is about the knowledge or about the English wording.
- **Paired comparisons** (the same questions before and after) with AuditKIT's bootstrap give the interval and p-value for "did it come back".
- **Attacks** test whether the knowledge is gone or only hidden: five worked examples in the prompt, an expert persona, and a short fine-tune on held-out questions.

## What you get

The model to ship is the CircuitKIT-layer one. In the Q4_K_M file it answers 37.1% of the hazardous-biology questions where the original answers 63.4%, and its general knowledge is 54.9% against the original's 54.8%. Across the other formats it costs up to 2.8 points. It behaves the same in the four other languages we tested, and prompt attacks do not bring the knowledge back. It is published as safetensors and as a Q4_K_M GGUF, next to the default-layer and 4-bit-aware versions.

## Cost

Tiny Aya's point is running offline. The table compares serving one million requests (400 tokens in, 150 out) through a closed API with running the unlearned 4-bit file yourself.

<!-- cost:start -->
| Option | Price or speed | Cost per 1M requests |
|---|---|---|
| Claude Haiku 5.5 API | $0.10 in / $0.50 out per 1M tokens | $115 |
| OpenAI gpt-6-luna API | $0.10 / $0.50 | $115 |
| Gemini 3.5 Flash-Lite API | $0.30 / $2.50 | $495 |
| OpenAI gpt-6.1-terra API | $1.00 / $5.00 | $1,150 |
| Claude Sonnet 5.5 API | $2.00 / $10.00 | $2,300 |
| Tiny Aya on one T4, AWS g4dn.xlarge | 149 output tokens/s measured (transformers fp16, batch 32), $0.526/hour | $147 (280 GPU-hours) |
| Tiny Aya on one T4, HF Inference Endpoints | same speed, $0.50/hour | $140 |
| Tiny Aya Q4_K_M on the user's own device | 14 output and 34 prompt tokens/s measured on 4 CPU threads (llama.cpp) | $0 per request, works offline |

Prices were read on 2026-11-09 from the providers' pricing pages and the AWS price list; speeds are our own runs ([`results/raw/bench_*.json`](results/raw/)). A served T4 costs about what the cheapest closed APIs cost and 8 to 16 times less than mid-tier ones. The case Tiny Aya is built for is the last row: it runs offline on the device, where no provider-side filter can step in, which is why the forgetting has to hold in the 4-bit file.
<!-- cost:end -->

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests                  # format round trips match llama.cpp bit for bit; every number in this README and the article is in the data
python scripts/build_results.py --fetch    # rebuild the results table and the Space data from the raw runs on Hugging Face
```

The runs themselves are Kaggle kernels on 2x T4, one script per step, in [`kaggle/`](kaggle/) with a table of what each produced. The edited weights of all 16 runs are in [tiny-aya-global-unlearn-bio-sweep](https://huggingface.co/itssaharsh/tiny-aya-global-unlearn-bio-sweep), and `scripts/patch_checkpoint.py` applies any of them to Tiny Aya Global. Unlearning uses the WMDP bio forget corpus, which CAIS gates; request access and it plugs in as described there. It is never redistributed here.

| Folder | What is in it |
|---|---|
| [`kaggle/`](kaggle/) | The scripts that produced every run, and `common.py`, the helpers they share |
| [`results/`](results/) | The raw runs (per model and format, per question, AuditKIT comparisons, CircuitKIT scores, tuning sweeps, prompt and fine-tuning attacks, speed tests) |
| [`scripts/`](scripts/) | Rebuild the results, curate the probe sets with CuratorKIT, build and publish the checkpoints and their cards |
| [`space/`](space/) | The Hugging Face Space: one static page that reads `data/results.json` |
| [`docs/ARTICLE.md`](docs/ARTICLE.md) | The write-up |
| [`tests/`](tests/) | Format round trips against llama.cpp; the page and write-up against the raw runs |

## What's real

| Part | Status |
|---|---|
| Unlearning, every score, every format, the CircuitKIT attribution, the AuditKIT statistics | measured on Tiny Aya Global |
| Q6_K and Q4_K_M | real llama.cpp files, published in the model collection |
| Q8_0, Q4_0, NF4 | bit-exact round trips of the decoder's linear weights, scored in PyTorch. The embedding table stays fp16, which matches bitsandbytes; a real llama.cpp Q8_0 or Q4_0 file also quantizes the embeddings |
| Translations | made by Tiny Aya Global itself and filtered by back-translation, not by a human translator |
| Prompt and fine-tuning attacks | measured on the 1,204 WMDP-bio questions not used as examples; prompts at fp16 and in the Q4_0 file, fine-tuning at fp16 |

Limits that change how to read the headline:

- One model. The two headline recipes have two seeds; the rest of the sweep has one.
- Settings were chosen on the same WMDP-bio and MMLU questions that are reported. The second seed is the only check outside that selection.
- Two layer windows were compared. CircuitKIT's faithfulness check ran on only 4 examples, so the layer choice is a lead that the unlearning results support, not a proof that attribution is the reason.
- Knowledge is measured by multiple choice, not by open generation.
- The forgetting does not survive fine-tuning.
- Costs are measured against the original in the same format. Running the same evaluation twice can differ by 1 to 3 questions, which is why a grid score and an AuditKIT score for the same model can differ by 0.1 to 0.2 points.

Built during the Lexsi AI Safety and Alignment Hackathon (November 2026), with Claude Code as the coding agent.

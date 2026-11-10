# Relapse: forget the hazard, keep the rest, even in the 4-bit file

You want a small model that runs offline on a phone and will not help anyone build a bioweapon. Unlearning can remove the dangerous knowledge, but two things usually go wrong. Nobody runs the checkpoint that was tested; people run a 4-bit copy of it. And unlearning costs some of everything else the model knows. Relapse measures both, in every file people actually download, using Cohere's Tiny Aya and Lexsi Labs' open-source safety stack, and finds a recipe where the 4-bit file keeps the forgetting and gives up nothing measurable for it.

**The result:** unlearning at the layers [CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT) found took Tiny Aya's hazardous-biology score from 63.4% to 37.1% in the llama.cpp Q4_K_M file, and general knowledge (MMLU) in that file stayed at 54.9% against the original's 54.8%. A second training seed agreed, and so did the whole sweep: across 16 unlearning runs, the CircuitKIT layers kept more general knowledge at every level of forgetting both reached. Among runs that got the hazard score to 40% or below, they cost 1.4 to 3.3 points in the 4-bit NF4 file; [SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s default layers cost 5.6 to 16.5. The forgetting held in all 7 formats, in Hindi, Arabic, Spanish, Swahili and reworded English, and against prompt attacks. It did not hold against fine-tuning, which is in the list of what didn't work below.

[![Relapse: the forget report flipping from the default layers to the CircuitKIT layers](docs/demo.gif)](https://huggingface.co/spaces/itssaharsh/relapse)

*35-second walk-through: [docs/demo.mp4](docs/demo.mp4)*

**Live demo:** [huggingface.co/spaces/itssaharsh/relapse](https://huggingface.co/spaces/itssaharsh/relapse) · **Write-up:** [the article](https://github.com/itssaharsh/relapse/blob/main/docs/ARTICLE.md) · **Models:** [collection](https://huggingface.co/collections/itssaharsh/relapse-unlearned-tiny-aya-models) · **Data and logs:** [collection](https://huggingface.co/collections/itssaharsh/relapse-runs-logs-and-probe-sets)

## Results

**What worked: unlearning at the layers CircuitKIT found.** We removed hazardous biology knowledge from [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B parameters, 70 languages) with SafeTune's RMU, scored on WMDP-bio (1,273 four-choice questions, 25% is guessing) and MMLU (1,000 questions). CircuitKIT's attribution showed the WMDP-bio answers running through the MLPs of layers 7 and 9, while RMU's default layers 5 and 6 carry little, so we put the edit at layers 7 to 9. Across two training seeds it took WMDP-bio from 63.3% to 36.8% and 45.6% at full precision, and to 37.1% and 45.3% in the Q4_K_M file. In the Q4_K_M file its general knowledge came out 0.1 and 0.6 points above the original's (AuditKIT p = 0.92 and 0.45), so no measurable cost. At fp16 it cost 2.2 points in one seed (p = 0.027) and nothing in the other (0.3 points above, p = 0.70).

**The comparison: SafeTune's default layers (5 to 7)** forgot about as much (38.7% and 44.4% at fp16) and cost general knowledge in every case: 3.8 and 3.7 points at fp16, 5.5 and 3.3 in Q4_K_M (AuditKIT p = 0.002, 0.006, 0.001, 0.013). Across both seeds and every file we measured, the CircuitKIT layers cost less.

**It is not one lucky setting.** We ran 8 RMU settings at each set of layers: learning rates, retain weights, steering strength, training through a 4-bit quantizer, and the second seed. Among the runs that got WMDP-bio to 40% or below, the default layers cost 3.8 to 19.9 points of general knowledge at fp16 and 5.6 to 16.5 in the NF4 file. The CircuitKIT layers cost 0.9 to 2.7 and 1.4 to 3.3. The strongest CircuitKIT run reached 31.0% for 2.7 points at fp16, and 30.1% for 3.3 in the NF4 file. The nearest default-layer run paid 8.1 points to reach 31.3% at fp16, and 12.4 for 28.6% in NF4.

![Every run: general knowledge kept against hazardous biology still answered, in the 4-bit NF4 file](docs/tradeoff.png)

**The forgetting held in every file, language and prompt we tried.** No recipe got its knowledge back in any of the 7 formats: the largest fp16-to-Q4_K_M move was +2.0 points (p = 0.12), and one was a further -3.4 (p = 0.004). Translated into Hindi, Arabic, Spanish and Swahili or reworded in English, the questions stayed forgotten: the original scores 46% to 57% per language, the CircuitKIT version 27% to 33%. Five worked examples in the prompt lifted the original to 63.5% but pushed the unlearned models down to 25.2% to 27.7%, and an expert persona left them at 32.2% to 35.0%.

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
- **Training through a 4-bit quantizer did nothing at the default layers** (38.9% against 38.7%, same MMLU). At the CircuitKIT layers it forgot a little more in both pairs we ran (34.0% against 36.8%, and 32.4% against 33.6%) at the same or lower cost. That is the "CircuitKIT + 4-bit aware" column.
- **4-bit compression did not bring the knowledge back** for any recipe we ran, so the question became what the 4-bit file pays for the forgetting.
- **The default layers' cost grew at 4-bit in one seed only.** In the first seed it went from 3.8 points at fp16 to 5.5 in Q4_K_M and 6.4 in NF4. In the second it stayed at 3.7, 3.3 and 3.0. What held in both seeds is the gap between the two recipes.
- **The forgetting did not survive fine-tuning.** 25 LoRA steps on 64 WMDP-bio questions with their answers brought the score on the other 1,204 questions back to 52.4% (default layers), 57.1% (CircuitKIT layers) and 55.5% (CircuitKIT, 4-bit aware), against 62.0% for the original. RMU hides the knowledge from prompts and from compression; it does not erase it, as [Łucki et al.](https://arxiv.org/abs/2409.18025) found on other models. Making it survive fine-tuning is the tamper-resistance problem, and the next thing to try.
- **How much a run forgets depends on the seed.** The same settings reached 36.8% in one seed and 45.6% in the other, which is why the results report both.

## For judges: one minute

1. Open the [Space](https://huggingface.co/spaces/itssaharsh/relapse). It opens on the CircuitKIT-layer model in the llama.cpp Q4_K_M file: still forgotten, nothing else lost, with the default layers' cost in the same file beside it.
2. Click **Default layers, same file**: the same forgetting now costs 5.5 points of general knowledge, and the page turns from green to red. **Ask in Swahili** shows another language.
3. Scroll once: the chart of all 16 runs, then the prompt and fine-tuning attacks.

Every number on the page comes from [`space/data/results.json`](space/data/results.json), which [`scripts/build_results.py`](scripts/build_results.py) rebuilds from the raw runs.

## How it works

Four Lexsi repositories in one pipeline, on a Cohere model, with every artifact on Hugging Face:

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

The model to ship is the CircuitKIT-layer one. Compared with the original Tiny Aya in the same Q4_K_M file, it answers 37.1% of the hazardous-biology questions where the original answers 63.4%, and it keeps the original's general knowledge (54.9% against 54.8%). It behaves the same in every other format and in the four other languages we tested, and prompt attacks do not bring the knowledge back. It is published as safetensors and as a Q4_K_M GGUF, next to the default-layer and 4-bit-aware versions.

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
| Q8_0, Q4_0, NF4 | bit-exact weight round trips scored in PyTorch, not llama.cpp's runtime kernels |
| Translations | made by Tiny Aya Global itself and filtered by back-translation, not by a human translator |
| Prompt and fine-tuning attacks | measured on the 1,204 WMDP-bio questions not used as examples; prompts at fp16 and in the Q4_0 file, fine-tuning at fp16 |

Limits that change how to read the headline: one model; two seeds for the two headline recipes and one for the rest of the sweep; knowledge measured by multiple choice, not by open generation; the forgetting does not survive fine-tuning; CircuitKIT's faithfulness check ran on only 4 examples, so the layer choice is a lead that the unlearning results confirmed, not a proof.

Built during the Lexsi AI Safety and Alignment Hackathon (November 2026), with Claude Code as the coding agent.

# Relapse: forget the hazard, keep the rest, even in the 4-bit file

Unlearning at the layers CircuitKIT pointed to took Tiny Aya's score on 1,273 hazardous-biology questions (WMDP-bio) from 63.4% to 37.1% in the 4-bit llama.cpp file, and kept 5.6 more points of general knowledge (MMLU, 1,000 questions) than the same unlearning at SafeTune's default layers (95% interval 2.8 to 8.3, p = 0.001). A second seed gave 3.9 more points (p = 0.006).

The setting: you want a small model that runs offline on a phone and will not help anyone build a bioweapon. Unlearning can remove that knowledge, but people run a 4-bit copy rather than the tested checkpoint, and unlearning costs some general knowledge. We measured both on Cohere Labs' [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B parameters, 70 languages) with four Lexsi Labs repos, and published every model, dataset and log on Hugging Face.

**What worked.** [CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT)'s attribution pointed to layers 7 to 9, and we put [SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s RMU edit there. It forgot as much as the default layers 5 to 7 and kept more general knowledge in all four head-to-head comparisons, significantly in three ([AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) paired bootstrap). Against the original model it cost -0.1 to 2.8 points depending on the file format; the default layers cost 3.7 to 6.4.

**Where it held, and where it didn't.** No recipe got its knowledge back in any of seven formats, in four other languages, with reworded questions, or with worked examples or an expert persona in the prompt. A short fine-tune did bring most of it back, which we report below with the other things that didn't work.

Try it: **[the Relapse Space](https://huggingface.co/spaces/itssaharsh/relapse)** · code and raw runs: **[github.com/itssaharsh/relapse](https://github.com/itssaharsh/relapse)**

![The forget report: every file format, the hazardous-biology score and the general-knowledge price](https://raw.githubusercontent.com/itssaharsh/relapse/main/space/og.png)

## Why ask

Tiny Aya is built to run offline, on phones and laptops, and Cohere ships it as GGUF files for exactly that. Offline there is no moderation API between the user and the model: whatever safety the model has lives in its weights. People also rarely run the checkpoint that was safety-tested. They run a quantized copy, and published work says that is exactly where unlearning can quietly fail. [Zhang et al. (ICLR 2025)](https://arxiv.org/abs/2410.16454) found that 4-bit quantization restored much of the knowledge gradient-based unlearning had removed, and Lexsi's own [Forgetting That Sticks](https://arxiv.org/abs/2605.15138) builds a method around the problem. Lexsi's safety stack is built around checking a fix in the form it ships in, so we ran that check for Tiny Aya across every format a user is likely to download.

## The results

**Head to head.** The same RMU edit at two sets of layers, compared on the same questions with AuditKIT's paired bootstrap:

<!-- recipes:start -->
| Seed | File | WMDP-bio: default / CircuitKIT layers | MMLU: default / CircuitKIT layers | MMLU kept extra (95% interval) | p |
|---|---|---|---|---|---|
| 1 | fp16 | 38.7% / 37.0% | 51.5% / 53.1% | +1.6 (-1.1 to +4.3) | 0.24 |
| 1 | Q4_K_M | 37.2% / 37.2% | 49.3% / 54.9% | +5.6 (+2.8 to +8.3) | 0.001 |
| 2 | fp16 | 44.5% / 45.6% | 51.7% / 55.7% | +4.0 (+1.3 to +6.6) | 0.003 |
| 2 | Q4_K_M | 41.1% / 45.4% | 51.5% / 55.4% | +3.9 (+1.3 to +6.6) | 0.006 |
<!-- recipes:end -->

The two models forgot the same amount in the first three rows (the hazard scores differ by 1.7 points or less, p = 0.33 or more). In the last row the default-layer model forgot 4.3 points more (p = 0.009).

**Every file format**, as hazardous biology still answered / general knowledge kept:

| File (WMDP-bio forget / MMLU keep) | Original | Default layers (5-7) | CircuitKIT layers (7-9) | CircuitKIT layers, 4-bit aware | Aggressive |
|---|---|---|---|---|---|
| fp16 | 63.3% / 55.4% | 38.7% / 51.6% | 36.8% / 53.2% | 34.0% / 54.5% | 25.4% / 43.5% |
| bf16 | 63.3% / 55.4% | 38.6% / 51.4% | 36.9% / 53.1% | 34.2% / 54.5% | 25.4% / 43.5% |
| Q8_0 | 63.2% / 55.1% | 38.4% / 51.4% | 37.3% / 53.2% | 34.6% / 54.4% | 26.4% / 43.8% |
| Q6_K | 63.2% / 55.6% | 39.1% / 51.7% | 36.4% / 53.8% | 34.9% / 53.9% | 26.9% / 42.9% |
| Q4_K_M | 63.4% / 54.8% | 37.2% / 49.3% | 37.1% / 54.9% | 35.3% / 54.5% | 27.3% / 40.7% |
| Q4_0 | 61.1% / 55.5% | 36.8% / 50.6% | 35.9% / 52.7% | 35.6% / 52.9% | 26.8% / 41.4% |
| NF4 | 61.1% / 52.7% | 37.2% / 46.3% | 35.3% / 51.2% | 34.7% / 50.8% | 26.7% / 36.2% |

The second-seed runs (fp16, Q8_0, Q4_K_M, Q4_0, NF4) are in the [results table](https://github.com/itssaharsh/relapse#results). Three things stand out:

1. **The forgetting holds in every file.** Across all 16 runs and every format, no hazard score rose by more than 2.0 points from its fp16 value (55 run-and-file pairs; the largest fall was 3.5). AuditKIT's paired fp16-to-Q4_K_M tests found no increase (the largest was +2.0 points, p = 0.12).
2. **Where you unlearn decides the price.** Against the original in the same file, the CircuitKIT-layer model cost -0.1 to 2.8 points across the seven formats in the first seed and -0.6 to 0.7 across five in the second. The default-layer model cost 3.7 to 6.4 and 3.0 to 3.7. In the Q4_K_M file the CircuitKIT-layer model's cost was not measurable in either seed (+0.1 points, 95% interval -1.9 to +2.0; +0.6, -1.1 to +2.3); at fp16 in the first seed it was 2.2 points (p = 0.027).
3. **Pushing harder at the default layers does not help.** The aggressive setting reached guessing (25.4%) but lost 11.9 points of general knowledge at fp16, 14.1 in Q4_K_M and 16.5 in NF4.

**Every run we made.** We ran 8 RMU settings at each set of layers. Among the runs that got WMDP-bio to 40% or below, the default layers cost 3.8 to 19.9 points of general knowledge at fp16, 4.9 to 14.1 in Q4_0 and 5.6 to 16.5 in NF4. The CircuitKIT layers cost 0.9 to 2.7, 2.5 to 3.5 and 1.4 to 3.3. The ranges stay apart for cut-offs from 35% to 42%; at 45% they overlap in Q4_0. The two sets were not tuned identically: the default layers got learning rates from 2e-5 to 3e-4 and one lower retain weight, the CircuitKIT layers got 1e-4 to 2e-4, one higher retain weight and one stronger steering setting, and runs used 250 or 300 batches.

![Every run: general knowledge kept against hazardous biology still answered, in the 4-bit NF4 file](https://raw.githubusercontent.com/itssaharsh/relapse/main/docs/tradeoff.png)

The forgetting carries across languages. On 707 WMDP-bio questions translated into Hindi, Arabic, Spanish and Swahili or reworded in English, the original model scores 46% to 57% per language at fp16. Unlearned at the default layers it scores 32% to 41%, and at the CircuitKIT layers 27% to 33%. The Q4_K_M files stay in the same range: 34% to 43% and 29% to 35%.

Pushing on the prompt did not bring it back either. With five solved WMDP-bio questions placed before each question, the original model went up to 63.5%, while the unlearned models went down to 25.2% to 27.7%: RMU scrambles the model's internal state when the hazardous topic is in context, and more of it scrambles more. With an expert-virologist persona they stayed at 32.2% to 35.0%. Both held in the 4-bit Q4_0 file too.

Fine-tuning is where it stopped holding. After 25 LoRA steps on 64 WMDP-bio questions with their answers, the score on the other 1,204 questions came back to 52.4% (default layers), 57.1% (CircuitKIT layers) and 55.5% (CircuitKIT, 4-bit aware), against 62.0% for the original. Before fine-tuning they scored 37.4%, 36.5% and 34.1% on those questions, so the CircuitKIT-layer models came back further. This suggests RMU hides the knowledge from prompts and from compression without erasing it, which is what [Łucki et al.](https://arxiv.org/abs/2409.18025) found on other models.

## The pipeline: four Lexsi repos on a Cohere model

**CuratorKIT builds the probe sets.** Tiny Aya Global translated a fixed sample of WMDP-bio questions into Hindi, Arabic, Spanish and Swahili and reworded them in English. We kept a translation only when its back-translation scored chrF 35 or more against the source. CuratorKIT's schema, exact and MinHash dedup and cleaning gates then curated the sets, with `strip_html` off so nothing in the questions is silently deleted. Each set ships with CuratorKIT's manifest and provenance file.

**CircuitKIT finds where the knowledge flows.** EAP-IG attribution on WMDP-bio questions over all 36 layers of Tiny Aya scored each layer's MLP and attention. RMU edits the MLPs of three consecutive early layers, and SafeTune's default is 5 to 7. Of all three-layer windows in layers 1 to 12, layers 7 to 9 have the most MLP attribution (0.30, against 0.23 for 5 to 7), so we tested that window. Other layers score higher (0, 16, 18 and 35), but they are outside the early range RMU is used in. The attribution ran on only 64 short questions and its faithfulness check is weak, so we treat it as a lead that the unlearning results support.

**SafeTune removes it.** RMU pushes the model's internal activations on hazardous text toward noise at one layer, while a retain loss on ordinary text (wikitext) keeps everything else in place. It edits only three MLP matrices. We needed real tuning for Tiny Aya: SafeTune's defaults changed nothing, a retain weight of 100 with learning rates 2e-5 to 2e-4 worked. We ran 8 settings at each set of layers and compare runs at similar forgetting. The 4-bit-aware variant trains the same objective with the edited weights randomly passed through llama.cpp's Q4_0/Q4_1 and NF4 quantizers (straight-through). SafeTune's NPO was also tried and was not selective on Tiny Aya: it either barely moved WMDP-bio or collapsed MMLU with it.

**AuditKIT scores every file.** Each question is scored by the log-likelihood of " A" to " D" after "Answer:", with no judge model, deterministic and comparable across formats. AuditKIT's paired bootstrap gives the "did it come back" and "what did it cost" intervals.

**The formats are made the way people make them.** We used the bf16 checkpoint, llama.cpp's own `convert_hf_to_gguf.py` and `llama-quantize` for Q6_K and Q4_K_M (dequantized back into PyTorch for scoring), and bit-exact re-implementations of llama.cpp's Q8_0/Q4_0 and of bitsandbytes NF4, tested against llama.cpp's reference code. All runs used free Kaggle T4s.

## What we tried that didn't work

These are the steps that led to the recipe above. All runs are in [the raw runs](https://github.com/itssaharsh/relapse/tree/main/results/raw).

- **SafeTune's RMU defaults barely moved Tiny Aya.** With retain weight 1200, learning rate 5e-5 and 150 batches, WMDP-bio went from 63.3% to 62.5%. A retain weight of 100 with 250 to 300 batches is what worked.
- **Pushing harder at the default layers forgot everything and broke the model.** At learning rate 1e-4 the score reached 25.4% (guessing), but MMLU fell to 43.5% and kept falling at 4-bit (36.2% in NF4). That is the "Aggressive" column.
- **SafeTune's NPO was not selective on Tiny Aya.** At learning rate 2e-4 it only reached 58.6%. At 5e-4 it reached 24.7% but took MMLU down to 23.4%, which is guessing too.
- **Training through a 4-bit quantizer did nothing at the default layers** (38.9% against 38.7%, same MMLU). At the CircuitKIT layers it forgot a little more in both pairs we ran (34.0% against 36.8%, and 32.4% against 33.6%). That is well inside the seed-to-seed spread, so we do not claim an effect. It is the "CircuitKIT + 4-bit aware" column.
- **4-bit compression did not bring the knowledge back** for any recipe we ran, so the question became what the 4-bit file pays for the forgetting.
- **The default layers' cost grew at 4-bit in one seed only.** In the first seed it went from 3.8 points at fp16 to 5.5 in Q4_K_M and 6.4 in NF4. In the second it stayed at 3.7, 3.3 and 3.0. What held in both seeds is the gap between the two recipes.
- **The forgetting did not survive fine-tuning.** 25 LoRA steps on 64 WMDP-bio questions with their answers brought the score on the other 1,204 questions back to 52.4% (default layers), 57.1% (CircuitKIT layers) and 55.5% (CircuitKIT, 4-bit aware), against 62.0% for the original. Before fine-tuning they scored 37.4%, 36.5% and 34.1% on those questions, so the CircuitKIT-layer models came back further. This suggests RMU hides the knowledge from prompts and from compression without erasing it, as [Łucki et al.](https://arxiv.org/abs/2409.18025) found on other models. Making it survive fine-tuning is the tamper-resistance problem, and the next thing to try.
- **How much a run forgets depends on the seed.** The same settings reached 36.8% in one seed and 45.6% in the other, which is why the results report both.

## Metrics and why

- **WMDP-bio accuracy** is the standard hazardous-knowledge measure in the unlearning literature, and the RMU paper's own. Multiple choice needs no judge.
- **MMLU accuracy** on a fixed seeded sample of 1,000 questions is the standard measure of what unlearning costs.
- **Translated and reworded probes** check that the model forgot the knowledge, not the English phrasing.
- **Paired statistics** compare the same questions in two conditions. That is the right test for "did this file bring anything back".
- **Attacks** test whether the knowledge is gone or only hidden: five worked examples in the prompt, an expert persona, and a short fine-tune on held-out questions.

## What it means

- **For anyone shipping an unlearned small model as GGUF:** on Tiny Aya, RMU-style unlearning survives every common format, other languages and prompt attacks. Measure the general-knowledge side in the shipped file too, because that is where recipes differed most.
- **For anyone relying on it:** this protects against people who can only prompt the shipped file. It does not protect against someone who fine-tunes the weights; that needs a tamper-resistant method, which is the next thing to try on this pipeline.
- **For Lexsi's stack:** the chain worked end to end on a new model family: CuratorKIT probes, CircuitKIT localization, SafeTune edits, AuditKIT statistics. The useful interaction was CircuitKIT suggesting *where* SafeTune edits. The window it suggested cost less at every level of forgetting both reached.
- **For Tiny Aya specifically:** forgetting done in English carried to Hindi, Arabic, Spanish and Swahili. That matters for a model whose point is 70 languages.

## Cost: self-hosting against closed APIs

For one million requests of 400 tokens in and 150 out:

| Option | Cost per 1M requests |
|---|---|
| Claude Haiku 5.5 or OpenAI gpt-6-luna API ($0.10/$0.50 per 1M tokens) | $115 |
| Gemini 3.5 Flash-Lite API ($0.30/$2.50) | $495 |
| OpenAI gpt-6.1-terra / Claude Sonnet 5.5 APIs | $1,150 / $2,300 |
| Tiny Aya on one T4 (149 output tokens/s measured, AWS $0.526/h) | $147 |
| Tiny Aya Q4_K_M on the user's own device (14 tokens/s measured on 4 CPU threads) | $0 per request, offline |

Prices were read on 2026-11-09. A served T4 costs roughly what the cheapest closed APIs cost. The reason to choose Tiny Aya is the last row, which is also where the safety has to sit in the weights.

## Limits

- One model. The two headline recipes have two seeds; the rest of the sweep has one, and all first-seed runs share the same data order.
- Settings were chosen on the same WMDP-bio and MMLU questions that are reported. The second seed is the only check outside that selection.
- Two layer windows were compared, so the results support CircuitKIT's suggestion without proving that attribution is the reason.
- The forgetting does not survive fine-tuning (above), and the CircuitKIT-layer models came back further.
- Costs are measured against the original in the same format. Running the same evaluation twice can differ by 1 to 3 questions.
- Knowledge is measured by multiple choice. A model can fail WMDP-bio questions and still say hazardous things in open generation; we did not measure generation.
- Q8_0, Q4_0 and NF4 are exact round trips of the decoder's linear weights scored in PyTorch; the embedding table stays fp16, where a real llama.cpp Q8_0 or Q4_0 file quantizes it too. Q6_K and Q4_K_M are real llama.cpp files.
- The translations are machine translations by Tiny Aya Global itself, filtered by back-translation, not checked by native speakers.

## Reproduce

```bash
git clone https://github.com/itssaharsh/relapse && cd relapse
pip install -r requirements.txt && python -m pytest -q tests   # round trips match llama.cpp; every number here is in the data
python scripts/build_results.py --fetch                     # rebuilds every number above from the raw runs
```

The Kaggle scripts that produced the runs, the unlearned models (safetensors and Q4_K_M GGUF), the per-question results, AuditKIT runs, CircuitKIT scores and CuratorKIT-gated probe sets are all public. The models are in the [Relapse models collection](https://huggingface.co/collections/itssaharsh/relapse-unlearned-tiny-aya-models) and the data in the [Relapse data collection](https://huggingface.co/collections/itssaharsh/relapse-runs-logs-and-probe-sets). The WMDP bio forget corpus is gated by CAIS and is not redistributed.

## Changelog

- 2026-11-10: first version.
- 2026-11-10, later: added the second seed, the 16-run sweep, the prompt attacks and the fine-tuning attack. The claim that the default layers' cost grows at 4-bit now says it held in one seed only.
- 2026-11-10, after an independent review: the headline is now the paired comparison between the two recipes, not the one file format with the lowest cost; the CircuitKIT attribution is described by what it shows (the best three-layer window, not 2 to 4 times more signal); the fine-tuning result says which models came back further; the limits name the selection on the reported questions.

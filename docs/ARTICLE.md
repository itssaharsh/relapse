# Relapse: forget the hazard, keep the rest, even in the 4-bit file

*Unlearning on Cohere's Tiny Aya with Lexsi Labs' open-source safety stack, measured in the files people actually download.*

You want a small model that runs offline on a phone and will not help anyone build a bioweapon. Unlearning can remove that knowledge, but two things usually go wrong: people run a 4-bit copy rather than the tested checkpoint, and unlearning costs some general knowledge. We measured both on [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B parameters, 70 languages) and found a recipe where the 4-bit file keeps the forgetting and gives up nothing measurable for it.

**The problem.** [SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s RMU at its default layers took Tiny Aya's WMDP-bio score (1,273 four-choice questions; 25% is guessing) from 63.3% to 38.7%. It stayed there in all seven formats we stored it in, from bf16 down to llama.cpp's Q4_K_M and bitsandbytes' NF4: 37.2% in Q4_K_M, an [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) paired bootstrap gives -1.5 points (95% interval -4.0 to +1.1, p = 0.24). But the general knowledge it cost grew in the 4-bit copy: 3.8 MMLU points at full precision, 5.5 in the Q4_K_M file (AuditKIT: p = 0.002 and 0.001).

**The fix.** [CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT)'s attribution pointed to different layers. The same edit there forgot slightly more (36.8%) and cost nothing measurable in the Q4_K_M file: MMLU 54.9% against the original's 54.8% (+0.1 points, p = 0.92). The forgetting held in every format and in Hindi, Arabic, Spanish, Swahili and reworded English.

Try it: **[the Relapse Space](https://huggingface.co/spaces/itssaharsh/relapse)** · code and raw runs: **[github.com/itssaharsh/relapse](https://github.com/itssaharsh/relapse)**

![The forget report: every file format, the hazardous-biology score and the general-knowledge price](https://raw.githubusercontent.com/itssaharsh/relapse/main/space/og.png)

## Why ask

Tiny Aya is built to run offline, on phones and laptops, and Cohere ships it as GGUF files for exactly that. Offline there is no moderation API between the user and the model: whatever safety the model has lives in its weights. People also rarely run the checkpoint that was safety-tested. They run a quantized copy, and published work says that is exactly where unlearning can quietly fail. [Zhang et al. (ICLR 2025)](https://arxiv.org/abs/2410.16454) found that 4-bit quantization restored much of the knowledge gradient-based unlearning had removed, and Lexsi's own [Forgetting That Sticks](https://arxiv.org/abs/2605.15138) builds a method around the problem. Lexsi's safety stack is built around checking a fix in the form it ships in, so we ran that check for Tiny Aya across every format a user is likely to download.

## The results

| File (WMDP-bio forget / MMLU keep) | Original | Default layers (5-7) | CircuitKIT layers (7-9) | CircuitKIT layers, 4-bit aware | Aggressive |
|---|---|---|---|---|---|
| fp16 | 63.3% / 55.4% | 38.7% / 51.6% | 36.8% / 53.2% | 34.0% / 54.5% | 25.4% / 43.5% |
| bf16 | 63.3% / 55.4% | 38.6% / 51.4% | 36.9% / 53.1% | 34.2% / 54.5% | 25.4% / 43.5% |
| Q8_0 | 63.2% / 55.1% | 38.4% / 51.4% | 37.3% / 53.2% | 34.6% / 54.4% | 26.4% / 43.8% |
| Q6_K | 63.2% / 55.6% | 39.1% / 51.7% | 36.4% / 53.8% | 34.9% / 53.9% | 26.9% / 42.9% |
| Q4_K_M | 63.4% / 54.8% | 37.2% / 49.3% | 37.1% / 54.9% | 35.3% / 54.5% | 27.3% / 40.7% |
| Q4_0 | 61.1% / 55.5% | 36.8% / 50.6% | 35.9% / 52.7% | 35.6% / 52.9% | 26.8% / 41.4% |
| NF4 | 61.1% / 52.7% | 37.2% / 46.3% | 35.3% / 51.2% | 34.7% / 50.8% | 26.7% / 36.2% |

Read each cell as "how much hazardous biology it still answers / how much general knowledge it keeps". Three things stand out:

1. **The forgetting holds in every file.** Every unlearned model stays within 2 points of its own full-precision WMDP-bio score in every file. None of AuditKIT's fp16-to-Q4_K_M comparisons is significant (p = 0.24, 0.89, 0.22, 0.12).
2. **At the default layers, the 4-bit file pays more.** There the general-knowledge gap to the original widens from 3.8 points (fp16) to 5.5 (Q4_K_M) and 6.4 (NF4). The aggressive setting shows it starkly, going from 11.9 points to 14.1 and 16.5.
3. **Where you unlearn decides the price.** At the CircuitKIT layers the gap is 2.2 points at fp16, +0.1 at Q4_K_M and 1.5 at NF4, while forgetting slightly more. Training the same edit through a 4-bit quantizer forgot the most (34.0%) with no detectable cost at fp16 or Q4_K_M (p = 0.42, 0.76).

The forgetting carries across languages. On 707 WMDP-bio questions translated into Hindi, Arabic, Spanish and Swahili or reworded in English, the original model scores 46% to 57% per language at fp16. Unlearned at the default layers it scores 32% to 41%, and at the CircuitKIT layers 27% to 33%. The Q4_K_M files stay in the same range: 34% to 43% and 29% to 35%.

## The pipeline: four Lexsi repos on a Cohere model

**CuratorKIT builds the probe sets.** Tiny Aya Global translated a fixed sample of WMDP-bio questions into Hindi, Arabic, Spanish and Swahili and reworded them in English. We kept a translation only when its back-translation scored chrF 35 or more against the source. CuratorKIT's schema, exact and MinHash dedup and cleaning gates then curated the sets, with `strip_html` off so nothing in the questions is silently deleted. Each set ships with CuratorKIT's manifest and provenance file.

**CircuitKIT finds where the knowledge flows.** EAP-IG attribution on WMDP-bio questions over all 36 layers of Tiny Aya scored each layer's MLP and attention. MLPs 7 and 9 carry two to four times more of the answer signal than 5 and 6. RMU's default edits layers 5 to 7, so we tested 7 to 9 instead. The attribution ran on only 64 short questions and its faithfulness check is weak, so we treat it as a lead. The unlearning results are what confirmed it.

**SafeTune removes it.** RMU pushes the model's internal activations on hazardous text toward noise at one layer, while a retain loss on ordinary text (wikitext) keeps everything else in place. It edits only three MLP matrices. We needed real tuning for Tiny Aya: SafeTune's defaults changed nothing, a retain weight of 100 with learning rates 2e-5 to 1.5e-4 worked, and we picked settings with similar forgetting so the comparison is fair. The 4-bit-aware variant trains the same objective with the edited weights randomly passed through llama.cpp's Q4_0/Q4_1 and NF4 quantizers (straight-through). SafeTune's NPO was also tried and was not selective on Tiny Aya: it either barely moved WMDP-bio or collapsed MMLU with it.

**AuditKIT scores every file.** Each question is scored by the log-likelihood of " A" to " D" after "Answer:", with no judge model, deterministic and comparable across formats. AuditKIT's paired bootstrap gives the "did it come back" and "what did it cost" intervals.

**The formats are made the way people make them.** We used the bf16 checkpoint, llama.cpp's own `convert_hf_to_gguf.py` and `llama-quantize` for Q6_K and Q4_K_M (dequantized back into PyTorch for scoring), and bit-exact re-implementations of llama.cpp's Q8_0/Q4_0 and of bitsandbytes NF4, tested against llama.cpp's reference code. All runs used free Kaggle T4s.

## What we tried that didn't work

These are the steps that led to the recipe above. All runs are in [the raw runs](https://github.com/itssaharsh/relapse/tree/main/results/raw).

- **SafeTune's RMU defaults barely moved Tiny Aya.** With retain weight 1200, learning rate 5e-5 and 150 batches, WMDP-bio went from 63.3% to 62.5%. A retain weight of 100 with 250 to 300 batches is what worked.
- **Pushing harder at the default layers forgot everything and broke the model.** At learning rate 1e-4 the score reached 25.4% (guessing), but MMLU fell to 43.5% and kept falling at 4-bit (36.2% in NF4). That is the "Aggressive" column.
- **SafeTune's NPO was not selective on Tiny Aya.** At learning rate 2e-4 it only reached 58.6%. At 5e-4 it reached 24.7% but took MMLU down to 23.4%, which is guessing too.
- **Training through a 4-bit quantizer did nothing at the default layers** (38.9% against 38.7%, same MMLU). At the CircuitKIT layers it forgot a little more, which is the "CircuitKIT + 4-bit aware" column.
- **4-bit compression did not bring the knowledge back** for any recipe we ran. So the question moved from "is it still gone?" to "what did the 4-bit file pay for it?", and that is where the default layers lost.

## Metrics and why

- **WMDP-bio accuracy** is the standard hazardous-knowledge measure in the unlearning literature, and the RMU paper's own. Multiple choice needs no judge.
- **MMLU accuracy** on a fixed seeded sample of 1,000 questions is the standard measure of what unlearning costs.
- **Translated and reworded probes** check that the model forgot the knowledge, not the English phrasing.
- **Paired statistics** compare the same questions in two conditions. That is the right test for "did this file bring anything back".

## What it means

- **For anyone shipping an unlearned small model as GGUF:** on Tiny Aya, RMU-style unlearning survives every common format. Test the general-knowledge side in the shipped file anyway. That is where the 4-bit copy differed.
- **For Lexsi's stack:** the chain worked end to end on a new model family: CuratorKIT probes, CircuitKIT localization, SafeTune edits, AuditKIT statistics. The useful interaction was CircuitKIT informing *where* SafeTune edits, which halved the price.
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

- One model, one seed per setting; the differences between the two CircuitKIT variants are within noise.
- Knowledge is measured by multiple choice. A model can fail WMDP-bio questions and still say hazardous things in open generation; we did not measure generation.
- Q8_0, Q4_0 and NF4 are exact weight round trips scored in PyTorch, not llama.cpp's runtime kernels; Q6_K and Q4_K_M are real llama.cpp files.
- The translations are machine translations by Tiny Aya Global itself, filtered by back-translation, not checked by native speakers.
- CircuitKIT's faithfulness check used 4 examples, so the layer choice rests on the unlearning results, not on the attribution alone.

## Reproduce

```bash
git clone https://github.com/itssaharsh/relapse && cd relapse
pip install -r requirements.txt && python -m pytest -q tests   # round trips match llama.cpp; every number here is in the data
python scripts/build_results.py --fetch                     # rebuilds every number above from the raw runs
```

The Kaggle scripts that produced the runs, the unlearned models (safetensors and Q4_K_M GGUF), the per-question results, AuditKIT runs, CircuitKIT scores and CuratorKIT-gated probe sets are all public. The models are in the [Relapse models collection](https://huggingface.co/collections/itssaharsh/relapse-unlearned-tiny-aya-models) and the data in the [Relapse data collection](https://huggingface.co/collections/itssaharsh/relapse-runs-logs-and-probe-sets). The WMDP bio forget corpus is gated by CAIS and is not redistributed.

## Changelog

- 2026-11-10: first version.

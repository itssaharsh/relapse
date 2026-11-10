# Relapse: is the forgotten knowledge still gone in the 4-bit file?

You made a model forget something dangerous. Then you shrank it to the 4-bit file people run on phones and laptops. Relapse checks whether the forgetting survived, using Cohere's Tiny Aya and Lexsi Labs' open-source safety stack.

**The answer, for Tiny Aya:** yes, the knowledge stays gone, in all 7 formats we tested and in 4 more languages. What changes at 4-bit is the price you pay for it in general knowledge, and *where* you unlearn decides that price.

[![Relapse: the forget report](space/og.png)](https://huggingface.co/spaces/itssaharsh/relapse)

**Live demo:** [huggingface.co/spaces/itssaharsh/relapse](https://huggingface.co/spaces/itssaharsh/relapse) · **Write-up:** [the article](https://github.com/itssaharsh/relapse/blob/main/docs/ARTICLE.md) · **Models:** [collection](https://huggingface.co/collections/itssaharsh/relapse-unlearned-tiny-aya-models) · **Data and logs:** [collection](https://huggingface.co/collections/itssaharsh/relapse-runs-logs-and-probe-sets)

## The answer

We removed hazardous biology knowledge from [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B parameters, 70 languages) with [SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s RMU. On the WMDP-bio benchmark (1,273 four-choice questions, 25% is guessing) the original model scores 63.3%. After unlearning it scores 38.7% at full precision, and every file we made from it stayed within 2 points of that: bf16, llama.cpp Q8_0, Q6_K, Q4_K_M and Q4_0, and bitsandbytes NF4. In the llama.cpp Q4_K_M file it scores 37.2% (AuditKIT paired bootstrap on the same 1,273 questions: -1.5 points, 95% interval -4.0 to +1.1, p = 0.24). Translated into Hindi, Arabic, Spanish and Swahili, or reworded in English, the questions were forgotten too.

What 4-bit did change was the cost. Unlearning at SafeTune's default layers (5 to 7) cost 3.8 points of general knowledge (MMLU, 1,000 questions) at fp16, 5.5 points in the Q4_K_M file and 6.4 in the NF4 file. CircuitKIT's attribution showed the WMDP-bio answers running through the MLPs of layers 7 and 9, so we moved the same edit to layers 7 to 9. That forgot slightly more (36.8%), cost 2.2 points at fp16, nothing measurable in the Q4_K_M file (+0.1 points, p = 0.92) and 1.5 points at NF4. Training that edit through a 4-bit quantizer forgot the most (34.0%) with no detectable cost at fp16 or Q4_K_M (p = 0.42 and 0.76); that is one seed, so read it as promising rather than settled.

<!-- results:start -->
| File (WMDP-bio / MMLU) | Original | Default layers | CircuitKIT layers | CircuitKIT + 4-bit aware | Aggressive |
|---|---|---|---|---|---|
| fp16 | 63.3% / 55.4% | 38.7% / 51.6% | 36.8% / 53.2% | 34.0% / 54.5% | 25.4% / 43.5% |
| bf16 | 63.3% / 55.4% | 38.6% / 51.4% | 36.9% / 53.1% | 34.2% / 54.5% | 25.4% / 43.5% |
| Q8_0 | 63.2% / 55.1% | 38.4% / 51.4% | 37.3% / 53.2% | 34.6% / 54.4% | 26.4% / 43.8% |
| Q6_K | 63.2% / 55.6% | 39.1% / 51.7% | 36.4% / 53.8% | 34.9% / 53.9% | 26.9% / 42.9% |
| Q4_K_M | 63.4% / 54.8% | 37.2% / 49.3% | 37.1% / 54.9% | 35.3% / 54.5% | 27.3% / 40.7% |
| Q4_0 | 61.1% / 55.5% | 36.8% / 50.6% | 35.9% / 52.7% | 35.6% / 52.9% | 26.8% / 41.4% |
| NF4 | 61.1% / 52.7% | 37.2% / 46.3% | 35.3% / 51.2% | 34.7% / 50.8% | 26.7% / 36.2% |
<!-- results:end -->

WMDP-bio accuracy is the knowledge that should be gone (lower is better; 25% is guessing). MMLU accuracy is the general knowledge that should stay (higher is better). Every cell is measured on the same questions; the Space shows the per-file details and the per-language scores.

## For judges: one minute

1. Open the [Space](https://huggingface.co/spaces/itssaharsh/relapse). It opens on the default-layer model in its 4-bit NF4 file: the hazardous biology is still gone, and the price in general knowledge is marked.
2. Click **CircuitKIT layers at 4-bit (NF4)**: same forgetting, a much smaller price, and the page turns from red to green.
3. Click **Llama.cpp Q4_K_M file** and **Ask in Swahili** to see the phone format and another language.

Every number on the page comes from [`space/data/results.json`](space/data/results.json), which [`scripts/build_results.py`](scripts/build_results.py) rebuilds from the raw runs.

## How it works

Four Lexsi repositories in one pipeline, on a Cohere model, with every artifact on Hugging Face:

| Step | Tool | What it did here |
|---|---|---|
| Build the probe sets | [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) | Schema, dedup and cleaning gates (with provenance manifests) on 707 WMDP-bio questions translated into Hindi, Arabic, Spanish and Swahili or reworded in English by Tiny Aya Global, kept only when the back-translation scored chrF 35 or more against the source |
| Find where the knowledge flows | [CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT) | EAP-IG attribution on WMDP-bio over Tiny Aya's 36 layers: MLPs 7 and 9 carry the answers, RMU's default layers 5 and 6 carry little, so the edit moved to layers 7 to 9 |
| Remove it | [SafeTune](https://github.com/Lexsi-Labs/SafeTune) | RMU on the WMDP bio forget corpus with a wikitext retain set; NPO was tried too and was not selective on Tiny Aya |
| Check every file | [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) | Scores each model in each format by answer log-likelihood and runs the paired bootstrap comparisons (fp16 against Q4_K_M, unlearned against original) |

The formats are made the way people make them: the bf16 checkpoint, llama.cpp's `convert_hf_to_gguf.py` and `llama-quantize` for Q6_K and Q4_K_M (read back into PyTorch for scoring), bit-exact re-implementations of llama.cpp's Q8_0 and Q4_0 and of bitsandbytes' NF4 for the rest. The quantization code is in [`relapse/`](relapse/) and tested against llama.cpp's own reference code in [`tests/`](tests/).

## Metrics, and why these

- **WMDP-bio accuracy** is the standard measure of hazardous biology knowledge in the unlearning literature (the RMU paper uses it). It is multiple choice, so it needs no judge model and gives the same answer every run.
- **MMLU accuracy** on a fixed sample of 1,000 questions is the standard measure of what unlearning costs.
- **Probability on the right answer** for both, a smoother signal than accuracy, is in the raw runs.
- **The same questions in four languages and reworded** test whether the forgetting is about the knowledge or about the English wording.
- **Paired comparisons** (the same questions before and after) with AuditKIT's bootstrap give the interval and p-value for "did it come back".

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
pip install -e ".[dev]" && python -m pytest -q tests      # precision round trips match llama.cpp bit for bit
python scripts/build_results.py --fetch                   # rebuild the results table and the Space data from the raw runs on Hugging Face
```

The runs themselves are Kaggle kernels on 2x T4, one script per step, in [`kaggle/`](kaggle/) with a table of what each produced. Unlearning uses the WMDP bio forget corpus, which CAIS gates; request access and it plugs in as described there. It is never redistributed here.

## What's real

| Part | Status |
|---|---|
| Unlearning, every score, every format, the CircuitKIT attribution, the AuditKIT statistics | measured on Tiny Aya Global |
| Q6_K and Q4_K_M | real llama.cpp files, published in the model collection |
| Q8_0, Q4_0, NF4 | bit-exact weight round trips scored in PyTorch, not llama.cpp's runtime kernels |
| Translations | made by Tiny Aya Global itself and filtered by back-translation, not by a human translator |

Limits that change how to read the headline: one model and one seed per setting; knowledge measured by multiple choice, not by open generation; CircuitKIT's faithfulness check ran on only 4 examples, so the layer choice is a lead that the unlearning results confirmed, not a proof.

Built during the Lexsi AI Safety and Alignment Hackathon (November 2026), with Claude Code as the coding agent.

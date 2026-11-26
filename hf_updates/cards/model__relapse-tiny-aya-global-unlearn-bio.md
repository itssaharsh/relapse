---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: transformers
tags:
- unlearning
- rmu
- wmdp
- gguf
---

# Tiny Aya Global, with hazardous biology unlearned

This is [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) after RMU unlearning of hazardous biology knowledge (the WMDP bio forget corpus), made with Lexsi Labs' [SafeTune](https://github.com/Lexsi-Labs/SafeTune). Three matrices are changed: the MLP output projections of layers 3 to 5. Everything else, including the tokenizer files, is the original.

**Paper:** [How Deep Does Unlearning Go? One Tokenizer Change Undoes It at Layers 5–7 but Not at Layers 3–5](https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf).

**Base model and tokenizer revision:** `CohereLabs/tiny-aya-global` at revision `af89d219b5` (2026-09-30), the commit that carries Cohere's tokenizer fix (`tokenizer_class: PreTrainedTokenizerFast`). This repository ships that revision's tokenizer files unchanged. Copies of Tiny Aya made before the fix carry an outdated setting that splits text differently under `transformers` v5. The checker [`scripts/check_tokenizer.py`](https://github.com/itssaharsh/relapse/blob/main/scripts/check_tokenizer.py), added with the paper, tells you which setting a repository, folder or GGUF file carries and what to do about it, e.g. `python scripts/check_tokenizer.py itssaharsh/relapse-tiny-aya-global-unlearn-bio`.

It is a research artifact of the [Relapse](https://github.com/itssaharsh/relapse) project, built for the Lexsi AI Safety and Alignment Hackathon. **It is not a safety guarantee.** A short fine-tune can bring the knowledge back (see below).

## Files

- `model.safetensors`: full precision (fp16).
- `relapse-tiny-aya-global-unlearn-bio-Q4_K_M.gguf`: a real llama.cpp Q4_K_M file of the same weights.
- `published_scores.json`: the scores of exactly these two files, measured after they were written.

## What was measured

All scores are accuracy on four-choice questions, read from the model's next-token probabilities. WMDP-bio is the hazard set (1,273 questions; lower is better; 25% is guessing). Everything was scored through the official tokenizer.

| Check | Original | This model |
|---|---|---|
| WMDP-bio (hazardous biology) | 63.4% | 37.3% |
| MMLU, 1,000 questions (general knowledge) | 55.4% | 54.5% |
| Benign biology (four MMLU biology subjects) | 62.5% | 54.9% |
| WMDP-bio with five worked examples in the prompt | 63.0% | 23.8% |
| WMDP-bio with an expert role-play prompt | 62.1% | 27.6% |
| Reworded English | 51.5% | 34.6% |
| Hindi | 50.8% | 34.8% |
| Swahili | 50.9% | 39.5% |
| Arabic | 50.0% | 32.3% |
| Spanish | 52.9% | 36.1% |

| The same weights, deployed | WMDP-bio |
|---|---|
| Full precision | 37.3% |
| Real Q4_0 file (llama.cpp) | 38.3% |
| Real Q4_K_M file (llama.cpp) | 37.9% |
| Through an outdated tokenizer setting | 36.5% |
| After 25 fine-tuning steps on 64 held-out hazard questions | 37.0% (from 37.5% on the same 1204 questions) |

## What to know before using it

- **The forgetting is broader than the hazard.** Benign biology drops from 62.5% to 54.9%.
- **Fine-tuning can undo it.** See the last row above. RMU hides knowledge more than it removes it.
- **Knowledge was measured by multiple choice**, not by open generation.
- **Check the file you ship, and the tokenizer you load.** In this project, a few unlearned runs regained 5 to 8 points in one 4-bit format while other formats of the same weights held (on Fire and Water, one exploratory run each). The files in this repository were each scored. Models unlearned through the outdated tokenizer setting at SafeTune's default layers regained 11.4 points on average once loaded through the official tokenizer; this model was unlearned through the official one, and through the outdated setting it scores as the table shows.
- A second run with another seed (global_W3-5_lr0.0002_s33) scored 47.8% on WMDP-bio and 53.7% on MMLU. This repository holds the run with the lower WMDP-bio.

## How it was made

SafeTune RMU, official tokenizer, layers 3 to 5, steering 3 × the activation norm, retain weight 100, 300 batches of 4, run `global_W3-5_lr0.0002_s34`. Layers 3 to 5 are the window that CircuitKIT's attribution ranks first when pooled over the five Tiny Aya models. Two registered tests compared them with SafeTune's default layers 5 to 7: they forgot more within 3 MMLU points in 9 of 10 comparisons (p = 0.011), and in 8 of 10 on fresh seeds (p = 0.055, short of our own line; the two exceptions are Tiny Aya Base). They cost more benign biology, about the same per hazard point forgotten. Unlearned through the outdated tokenizer setting, default-layer runs relapsed through the official tokenizer in 16 of 16 runs that count and layers 3 to 5 in 0 of 16, on the four post-trained variants; Tiny Aya Base rose at every window (8.5 to 11.6 points at layers 3 to 5). The pre-registrations, every run's scores and the paper are in the [repository](https://github.com/itssaharsh/relapse).

## License

CC-BY-NC-4.0, the license of the original model. Use is also subject to Cohere's terms for Tiny Aya.

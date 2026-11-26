---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-base
library_name: transformers
tags:
- unlearning
- rmu
- wmdp
- gguf
---

# Tiny Aya Base, with hazardous biology unlearned

This is [CohereLabs/tiny-aya-base](https://huggingface.co/CohereLabs/tiny-aya-base) after RMU unlearning of hazardous biology knowledge (the WMDP bio forget corpus), made with Lexsi Labs' [SafeTune](https://github.com/Lexsi-Labs/SafeTune). Three matrices are changed: the MLP output projections of layers 3 to 5. Everything else, including the tokenizer files, is the original.

It is a research artifact of the [Relapse](https://github.com/itssaharsh/relapse) project, built for the Lexsi AI Safety and Alignment Hackathon. **It is not a safety guarantee.** A short fine-tune can bring the knowledge back (see below).

## Files

- `model.safetensors`: full precision (fp16).
- `relapse-tiny-aya-base-unlearn-bio-Q4_K_M.gguf`: a real llama.cpp Q4_K_M file of the same weights.
- `published_scores.json`: the scores of exactly these two files, measured after they were written.

## What was measured

All scores are accuracy on four-choice questions, read from the model's next-token probabilities. WMDP-bio is the hazard set (1,273 questions; lower is better; 25% is guessing). Everything was scored through the official tokenizer.

| Check | Original | This model |
|---|---|---|
| WMDP-bio (hazardous biology) | 60.1% | 33.2% |
| MMLU, 1,000 questions (general knowledge) | 57.6% | 56.4% |
| Benign biology (four MMLU biology subjects) | 65.4% | 57.5% |
| WMDP-bio with five worked examples in the prompt | 65.4% | 24.4% |
| WMDP-bio with an expert role-play prompt | 64.3% | 30.0% |
| Reworded English | 48.5% | 32.7% |
| Hindi | 49.7% | 35.8% |
| Swahili | 50.0% | 37.7% |
| Arabic | 46.2% | 30.1% |
| Spanish | 52.9% | 26.9% |

| The same weights, deployed | WMDP-bio |
|---|---|
| Full precision | 33.2% |
| Real Q4_0 file (llama.cpp) | 33.1% |
| Real Q4_K_M file (llama.cpp) | 33.7% |
| Through an outdated tokenizer setting | 34.3% |
| After 25 fine-tuning steps on 64 held-out hazard questions | 52.1% (from 32.7% on the same 1204 questions) |

## What to know before using it

- **The forgetting is broader than the hazard.** Benign biology drops from 65.4% to 57.5%.
- **Fine-tuning can undo it.** See the last row above. RMU hides knowledge more than it removes it.
- **Knowledge was measured by multiple choice**, not by open generation.
- **Check the file you ship.** In this project, some unlearned models regained 5 to 8 points in one 4-bit format while other formats of the same weights held. The files in this repository were each scored.
- A second run with another seed (base_W3-5_lr6e-05_s33) scored 39.2% on WMDP-bio and 56.8% on MMLU. This repository holds the run with the lower WMDP-bio.

## How it was made

SafeTune RMU, official tokenizer, layers 3 to 5, steering 3 × the activation norm, retain weight 100, 300 batches of 4, run `base_W3-5_lr6e-05_s34`. Layers 3 to 5 are the window that CircuitKIT's attribution ranks first when pooled over the five Tiny Aya models; in pre-registered tests they forgot more than SafeTune's default layers at the same cost on four of the five models. The pre-registrations, every run's scores and the paper are in the [repository](https://github.com/itssaharsh/relapse).

## License

CC-BY-NC-4.0, the license of the original model. Use is also subject to Cohere's terms for Tiny Aya.

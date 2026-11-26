---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-fire
library_name: transformers
tags:
- unlearning
- rmu
- wmdp
- gguf
---

# Tiny Aya Fire, with hazardous biology unlearned

This is [CohereLabs/tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire) after RMU unlearning of hazardous biology knowledge (the WMDP bio forget corpus), made with Lexsi Labs' [SafeTune](https://github.com/Lexsi-Labs/SafeTune). Three matrices are changed: the MLP output projections of layers 3 to 5. Everything else, including the tokenizer files, is the original.

It is a research artifact of the [Relapse](https://github.com/itssaharsh/relapse) project, built for the Lexsi AI Safety and Alignment Hackathon. **It is not a safety guarantee.** A short fine-tune can bring the knowledge back (see below).

## Files

- `model.safetensors`: full precision (fp16).
- `relapse-tiny-aya-fire-unlearn-bio-Q4_K_M.gguf`: a real llama.cpp Q4_K_M file of the same weights.
- `published_scores.json`: the scores of exactly these two files, measured after they were written.

## What was measured

All scores are accuracy on four-choice questions, read from the model's next-token probabilities. WMDP-bio is the hazard set (1,273 questions; lower is better; 25% is guessing). Everything was scored through the official tokenizer.

| Check | Original | This model |
|---|---|---|
| WMDP-bio (hazardous biology) | 63.5% | 40.5% |
| MMLU, 1,000 questions (general knowledge) | 57.9% | 56.9% |
| Benign biology (four MMLU biology subjects) | 64.3% | 58.3% |
| WMDP-bio with five worked examples in the prompt | 66.4% | 23.8% |
| WMDP-bio with an expert role-play prompt | 63.3% | 36.0% |
| Reworded English | 51.5% | 30.7% |
| Hindi | 55.6% | 38.0% |
| Swahili | 51.7% | 40.4% |
| Arabic | 49.5% | 33.3% |
| Spanish | 52.9% | 35.3% |

| The same weights, deployed | WMDP-bio |
|---|---|
| Full precision | 40.5% |
| Real Q4_0 file (llama.cpp) | 37.9% |
| Real Q4_K_M file (llama.cpp) | 38.3% |
| Through an outdated tokenizer setting | 38.8% |
| After 25 fine-tuning steps on 64 held-out hazard questions | 50.2% (from 40.5% on the same 1204 questions) |

## What to know before using it

- **The forgetting is broader than the hazard.** Benign biology drops from 64.3% to 58.3%.
- **Fine-tuning can undo it.** See the last row above. RMU hides knowledge more than it removes it.
- **Knowledge was measured by multiple choice**, not by open generation.
- **Check the file you ship.** In this project, some unlearned models regained 5 to 8 points in one 4-bit format while other formats of the same weights held. The files in this repository were each scored.
- A second run with another seed (fire_W3-5_lr6e-05_s33) scored 42.8% on WMDP-bio and 55.6% on MMLU. This repository holds the run with the lower WMDP-bio.

## How it was made

SafeTune RMU, official tokenizer, layers 3 to 5, steering 3 × the activation norm, retain weight 100, 300 batches of 4, run `fire_W3-5_lr6e-05_s34`. Layers 3 to 5 are the window that CircuitKIT's attribution ranks first when pooled over the five Tiny Aya models; in pre-registered tests they forgot more than SafeTune's default layers at the same cost on four of the five models. The pre-registrations, every run's scores and the paper are in the [repository](https://github.com/itssaharsh/relapse).

## License

CC-BY-NC-4.0, the license of the original model. Use is also subject to Cohere's terms for Tiny Aya.

---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: transformers
pipeline_tag: text-generation
tags: [unlearning, safety, rmu, safetune, wmdp, quantization, gguf, tiny-aya, lexsi]
datasets: [cais/wmdp, cais/mmlu]
---

# Tiny Aya Global, hazardous biology unlearned (CircuitKIT + 4-bit aware)

> **Superseded Stage 1 artifact (status on 27 November 2026).** This repository was made in the first round of the Lexsi AI Safety and Alignment Hackathon, on Kaggle, through an outdated tokenizer setting that splits text differently from the way Tiny Aya was trained. Every Stage 1 score and claim made from it is withdrawn, including the 4-bit relapse at layers 10 to 12 and its repair. Until 26 November 2026 (UTC) its `tokenizer_config.json` carried that setting (`tokenizer_class: CohereTokenizerFast`); on that date it was replaced with Cohere's fixed file (from `CohereLabs/tiny-aya-global` at `af89d219b5`), so the repository now loads with the official tokenizer. The Stage 1 results were made with revision `0509cef6ef` of this repository, which keeps the outdated setting. Use the Stage 2 models instead: [Global](https://huggingface.co/itssaharsh/relapse-tiny-aya-global-unlearn-bio), [Fire](https://huggingface.co/itssaharsh/relapse-tiny-aya-fire-unlearn-bio), [Earth](https://huggingface.co/itssaharsh/relapse-tiny-aya-earth-unlearn-bio), [Water](https://huggingface.co/itssaharsh/relapse-tiny-aya-water-unlearn-bio), [Base](https://huggingface.co/itssaharsh/relapse-tiny-aya-base-unlearn-bio).

**What the final paper found** ([An Outdated Tokenizer Setting Overstated Unlearning Applied at Layers 5–7 in 16 of 16 Runs, but Not at Layers 3–5 in Post-Trained Models](https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf)). Runs unlearned and audited through an outdated tokenizer setting at SafeTune's default layers (5 to 7) regained 11.4 points on average once loaded through the official tokenizer, in 16 of 16 runs that count; those runs had also lost 5.8 MMLU points where they were audited and 1.7 once deployed, and with single-line prompts none of them relapsed. At layers 3 to 5, 0 of 16 did, on the four post-trained variants; Tiny Aya Base rose at every window. In 4-bit files the forgetting held in 107 of 109 effective runs. Of 25 registered hypotheses, 11 held, 2 held in part, 11 failed and 1 could not be tested.

## What this repository holds

- Tiny Aya Global after SafeTune RMU unlearning of hazardous biology: layers 7 to 9, trained through a 4-bit quantizer. Trained and first scored through the outdated tokenizer setting.
- Files: `config.json`, `generation_config.json`, `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`, `model.safetensors.index.json`, `special_tokens_map.json`, `tiny-aya-global-unlearn-bio-circuitkit-4bit-aware-Q4_K_M.gguf`, `tokenizer.json`, `tokenizer_config.json`.

## Base model and tokenizer revision

Base model: [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global). The Stage 1 runs used the copy Kaggle Models held (`cohere-labs/tiny-aya/transformers/global/1`, uploaded 16 February 2026), which Kaggle identifies only as version 1 and which carries the outdated setting (`tokenizer_class: CohereTokenizerFast`). Cohere fixed the setting on Hugging Face on 30 September 2026 (revision `af89d219b5` of the official repository).

The checker [`scripts/check_tokenizer.py`](https://github.com/itssaharsh/relapse/blob/main/scripts/check_tokenizer.py), added with the paper, reports which setting a repository, folder or GGUF file carries and what to do about it, reading only its tokenizer file or GGUF header: `python scripts/check_tokenizer.py itssaharsh/tiny-aya-global-unlearn-bio-circuitkit-4bit-aware`.

The Stage 1 card of this repository, with its withdrawn description and scores, is kept at [`hf_updates/previous/`](https://github.com/itssaharsh/relapse/tree/main/hf_updates/previous), and the Stage 1 write-up at [`docs/archive/`](https://github.com/itssaharsh/relapse/tree/main/docs/archive).

## License

CC-BY-NC-4.0, the license of the original model. Use is also subject to Cohere's terms for Tiny Aya.

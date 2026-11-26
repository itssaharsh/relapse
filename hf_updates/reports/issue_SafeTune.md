TITLE: RMU scored through a stale Tiny Aya tokenizer setting overstates forgetting; the run does not record which tokenizer made its batches

**Setting.** SafeTune 0.1.13, `rmu_unlearn` with `RMUConfig`, on Cohere's five Tiny Aya models (3.35B), WMDP-bio as the forget measure and MMLU as the retain measure. Default window: `layer_id=7`, `update_layer_ids=[5, 6, 7]`.

**What we measured.** Copies of Tiny Aya made before 30 September 2026 name `CohereTokenizerFast` (or `CohereTokenizer`) in `tokenizer_config.json`; under `transformers` 5.x that class splits text differently from the tokenizer the model was trained with (digits one by one, punctuation separated from the word it joins, no joined punctuation-plus-line-break pieces). Cohere changed the field on Hugging Face on that date; the copy on Kaggle Models still has the old value.

In a pre-registered test, runs unlearned and scored through the old setting at the default window scored 39.4% on WMDP-bio there and 50.8% through the official tokenizer: 16 of 16 runs rose by 5 points or more. The same runs had lost 5.8 MMLU points where they were scored and 1.7 through the official tokenizer. At layers 3 to 5, 0 of 16 runs rose, on the four post-trained variants; Tiny Aya Base rose at every window. Scored again with every prompt on a single line (one layout without line breaks, options joined by ` | `), 0 of those 16 runs rose by 5 points, although the two settings still split 815 of the 1,273 single-line prompts differently; so the effect was absent in the one layout without line breaks that we tested. We cannot say the layer is the reason: across 120 runs, those that had lost 3 or more MMLU points in the old pipeline rose in 72 of 72 cases and the others in 0 of 48, and each window had its own learning rates.

**Why it touches SafeTune.** `rmu_unlearn` takes tokenized batches and returns edited weights. Nothing in the run records which tokenizer produced the batches, so two runs that differ only in the tokenizer setting cannot be told apart afterwards.

**What a maintainer could change.** Record the tokenizer class and a hash of the tokenizer files with the run (config or provenance). Optionally warn when the retain measure falls by several points in the same pipeline the forget measure was read in.

Paper: https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf (Section 4.1). A checker for the setting, which reads only tokenizer files: https://github.com/itssaharsh/relapse/blob/main/scripts/check_tokenizer.py. Per-question results: https://github.com/itssaharsh/relapse/tree/main/results/shift2

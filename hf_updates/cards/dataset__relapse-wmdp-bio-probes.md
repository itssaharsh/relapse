---
license: cc-by-nc-4.0
language: [en, hi, ar, es, sw]
task_categories: [multiple-choice]
tags: [wmdp, unlearning, multilingual, safety, curatorkit, tiny-aya, relapse]
pretty_name: Relapse WMDP-bio probes (translated and reworded)
configs:
- config_name: default
  data_files: probes.json
---

# Relapse WMDP-bio probes

<!-- relapse-status:start -->
> **Status on 27 November 2026.** The translations and rewordings were made in Stage 1 by Tiny Aya Global through Kaggle's copy, so through the outdated tokenizer setting; no native speaker checked them. The paper's full check scores the released models on them through the official tokenizer.
>
> **What the final paper found** ([An Outdated Tokenizer Setting Overstated Unlearning Applied at Layers 5–7 in 16 of 16 Runs, but Not at Layers 3–5 in Post-Trained Models](https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf)). Runs unlearned and audited through an outdated tokenizer setting at SafeTune's default layers (5 to 7) regained 11.4 points on average once loaded through the official tokenizer, in 16 of 16 runs that count; those runs had also lost 5.8 MMLU points where they were audited and 1.7 once deployed, and with single-line prompts none of them relapsed. At layers 3 to 5, 0 of 16 did, on the four post-trained variants; Tiny Aya Base rose at every window. In 4-bit files the forgetting held in 107 of 109 effective runs. Of 25 registered hypotheses, 11 held, 2 held in part, 11 failed and 1 could not be tested.
>
> **Base model and tokenizer revision.** Every Stage 2 job downloaded the main branch of `CohereLabs/tiny-aya-<variant>` when it started; by the job start times and the repositories' history that was the revision carrying Cohere's tokenizer fix (Global `af89d219b5`, Fire `73500c80fc`, Earth `0624fafbe2`, Water `41a99216d1`, Base `d62ac17e24`, all committed 30 September 2026). Stage 1 used the copy Kaggle Models held, which carries the outdated setting.
>
> **Tokenizer checker.** [`scripts/check_tokenizer.py`](https://github.com/itssaharsh/relapse/blob/main/scripts/check_tokenizer.py), added with the paper, tells you whether a copy of Tiny Aya carries the outdated tokenizer setting and what to do about it.
<!-- relapse-status:end -->

707 questions from the public [WMDP-bio](https://huggingface.co/datasets/cais/wmdp) benchmark, used in
[Relapse](https://huggingface.co/spaces/itssaharsh/relapse) to check that unlearning removed the knowledge,
not just the English phrasing.

| Set | Questions |
|---|---|
| Hindi | 187 |
| Arabic | 186 |
| Spanish | 119 |
| Swahili | 114 |
| English, reworded | 101 |

**How they were made.** A fixed sample of 200 WMDP-bio test questions (seed 0) was translated, and separately reworded, by
[CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global). A translation was kept only when its
back-translation into English scored chrF 35 or more against the source. Then [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT)'s
schema gate, exact and MinHash dedup and text cleaner curated each set (`strip_html` off). Each language's CuratorKIT
manifest, provenance file and dataset card are in `curatorkit/<lang>/`. `curation_report.json` has the counts.

**Fields.** `source_id` (the question's index in `cais/wmdp` `wmdp-bio` test), `lang`, `kind` (translation or paraphrase),
`item` (question and options A to D), `answer` (index of the correct option, unchanged from the source),
`backtranslation_chrf`. `source_en.json` holds the English source questions for the sample.

**Limits.** Machine translations, not checked by native speakers. The questions are WMDP's proxy questions, which are
public and designed not to be directly hazardous. Code: [github.com/itssaharsh/relapse](https://github.com/itssaharsh/relapse) (`scripts/curate_probes.py`).

**License.** CC-BY-NC-4.0: the translations and rewordings were written by Tiny Aya Global, which is released under that license. The source questions are from WMDP (MIT).

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

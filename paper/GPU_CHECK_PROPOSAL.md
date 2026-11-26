# Optional GPU check (proposal; nothing has been run)

Not submitted. To be registered and committed as `results/PREREGISTRATION_6.md` before any job starts, only if the author says yes.

**What.** Rescore the 60 saved runs of pre-registration 5 (and the five originals) with every prompt on a single line: the same question and the four options joined by " | " in place of line breaks, ending in "Answer:". fp16, WMDP-bio (1,273) and MMLU (1,000), through both tokenizer settings. Scoring only: no training, no new seeds. The panel ranked this above the fine-tuning attack on the default layers, because it is the one check that speaks to what the two settings do to the text.

**Registration text.** Definitions as in pre-registration 5 (rise, effective run by the multiple-choice audit, relapse). S1: among the 16 effective default-layer runs, the rise with single-line prompts is smaller than with the multiple-choice layout in more runs than not (one-sided sign test, alpha 0.05). S2: at least half of those runs still rise by 5 points or more with paired p < 0.05. S3 (control): at layers 3 to 5, fewer than one in five effective runs relapse. Tiny Aya Base is reported whatever range it lands in. Every revision is pinned (the model revisions of Table "revisions" in the appendix, the llama.cpp commit, this repository's commit). The result is reported whichever way it falls, in the paper's Section 4.1 and Limitations.

**Cost and time.** Ten jobs of about 20 minutes on one L4 (about 3.5 GPU-hours, about $3) from the finalist credits already granted, of which about $60 remain by the paper's own cost note, or the free weekly Kaggle T4 quota. About 2 hours end to end with analysis and the paper edit; the edit must fit inside the page limit.

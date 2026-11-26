---
title: Relapse
emoji: ⬛
colorFrom: red
colorTo: green
sdk: static
pinned: false
license: mit
short_description: Does unlearning survive 4-bit files and the tokenizer?
---

# Relapse: does unlearning survive deployment?

<!-- relapse-status:start -->
> **Status on 26 November 2026.** This page shows the final, pre-registered Stage 2 results; every number on it is read from the result files.
>
> **What the final paper found** ([An Outdated Tokenizer Setting Overstated Unlearning Applied at Layers 5–7 in 16 of 16 Runs, but Not at Layers 3–5](https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf)). Runs unlearned and audited through an outdated tokenizer setting at SafeTune's default layers (5 to 7) regained 11.4 points on average once loaded through the official tokenizer, in 16 of 16 runs that count; those runs had also lost 5.8 MMLU points where they were audited and 1.7 once deployed, and with single-line prompts none of them relapsed. At layers 3 to 5, 0 of 16 did, on the four post-trained variants; Tiny Aya Base rose at every window. In 4-bit files the forgetting held in 107 of 109 runs. Of 25 registered hypotheses, 11 held, 2 held in part, 11 failed and 1 could not be tested.
>
> **Base model and tokenizer revision.** Every Stage 2 job downloaded the main branch of `CohereLabs/tiny-aya-<variant>` when it started; by the job start times and the repositories' history that was the revision carrying Cohere's tokenizer fix (Global `af89d219b5`, Fire `73500c80fc`, Earth `0624fafbe2`, Water `41a99216d1`, Base `d62ac17e24`, all committed 30 September 2026). Stage 1 used the copy Kaggle Models held, which carries the outdated setting.
>
> **Tokenizer checker.** [`scripts/check_tokenizer.py`](https://github.com/itssaharsh/relapse/blob/main/scripts/check_tokenizer.py), added with the paper, tells you whether a copy of Tiny Aya carries the outdated tokenizer setting and what to do about it.
<!-- relapse-status:end -->

Cohere's Tiny Aya models, made to forget hazardous biology (WMDP-bio) with Lexsi Labs' SafeTune, then checked in real 4-bit files and through the official tokenizer. These are the final, pre-registered Stage 2 results; the Stage 1 claims were withdrawn ([the correction](https://github.com/itssaharsh/relapse#correction-november-21-2026)).

The page is plain HTML, CSS and JavaScript with no build step. Every number on it is read from `data/stage2.json`, which `scripts/build_space.py` writes from the result files in the [repository](https://github.com/itssaharsh/relapse) (`python scripts/build_space.py --check` also checks the shared figures against the README). `data/results.json` holds the Stage 1 models rescored; the page no longer reads it, but the repository's tests still do.

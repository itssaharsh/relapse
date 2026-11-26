---
title: "Relapse: the write-up"
emoji: 📄
colorFrom: green
colorTo: red
sdk: static
pinned: false
license: mit
short_description: Forget the hazard, keep the rest, even in the 4-bit file
---

<!-- relapse-status:start -->
> **Status on 26 November 2026.** This page is the write-up of the final results, rebuilt from `docs/ARTICLE.md`.
>
> **What the final paper found** ([How Deep Does Unlearning Go? One Tokenizer Change Undoes It at Layers 5–7 but Not at Layers 3–5](https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf)). Runs unlearned and audited through an outdated tokenizer setting at SafeTune's default layers (5 to 7) regained 11.4 points on average once loaded through the official tokenizer, in 16 of 16 runs that count. At layers 3 to 5, 0 of 16 did, on the four post-trained variants; Tiny Aya Base rose at every window. In 4-bit files the forgetting held in 107 of 109 runs. Of 22 registered hypotheses, 9 held, 2 held in part, 10 failed and 1 could not be tested.
>
> **Base model and tokenizer revision.** Every Stage 2 job downloaded the main branch of `CohereLabs/tiny-aya-<variant>` when it started; by the job start times and the repositories' history that was the revision carrying Cohere's tokenizer fix (Global `af89d219b5`, Fire `73500c80fc`, Earth `0624fafbe2`, Water `41a99216d1`, Base `d62ac17e24`, all committed 30 September 2026). Stage 1 used the copy Kaggle Models held, which carries the outdated setting.
>
> **New:** [`scripts/check_tokenizer.py`](https://github.com/itssaharsh/relapse/blob/main/scripts/check_tokenizer.py) tells you whether a copy of Tiny Aya carries the outdated tokenizer setting.
<!-- relapse-status:end -->

The write-up for [Relapse](https://huggingface.co/spaces/itssaharsh/relapse), a Lexsi AI Safety and Alignment Hackathon entry.
The same text is at [github.com/itssaharsh/relapse](https://github.com/itssaharsh/relapse/blob/main/docs/ARTICLE.md); this page is rebuilt from it with `scripts/build_article_space.py`.

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

Cohere's Tiny Aya models, made to forget hazardous biology (WMDP-bio) with Lexsi Labs' SafeTune, then checked in real 4-bit files and through the official tokenizer. These are the final, pre-registered Stage 2 results; the Stage 1 claims were withdrawn ([the correction](https://github.com/itssaharsh/relapse#correction-november-21-2026)).

The page is plain HTML, CSS and JavaScript with no build step. Every number on it is read from `data/stage2.json`, which `scripts/build_space.py` writes from the result files in the [repository](https://github.com/itssaharsh/relapse) (`python scripts/build_space.py --check` also checks the shared figures against the README). `data/results.json` holds the Stage 1 models rescored; the page no longer reads it, but the repository's tests still do.

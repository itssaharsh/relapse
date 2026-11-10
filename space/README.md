---
title: Relapse
emoji: ⬛
colorFrom: red
colorTo: green
sdk: static
pinned: false
license: mit
short_description: Is the forgotten knowledge still gone in the 4-bit file?
---

# Relapse: is the forgotten knowledge still gone in the 4-bit file?

Cohere's Tiny Aya Global, made to forget hazardous biology (WMDP-bio) with Lexsi Labs' SafeTune, then re-tested in the files people download (fp16, bf16, Q8_0, Q6_K, Q4_K_M, Q4_0, NF4) and in Hindi, Arabic, Spanish and Swahili. The forgetting holds in every file; the general knowledge it costs grows at 4-bit at SafeTune's default layers and stays flat at the layers CircuitKIT found.

Everything on the page is read from `data/results.json`, rebuilt from the raw runs by `scripts/build_results.py` in [github.com/itssaharsh/relapse](https://github.com/itssaharsh/relapse).

TITLE: EAP-IG layer ranking on the built-in wmdp-bio task changes with the number of prompts on 3 of 5 Tiny Aya models

**Setting.** CircuitKIT 0.2.1, EAP-IG on the built-in WMDP task (`wmdp-bio`, questions of at most 80 tokens, 5 integration steps, heads and MLPs, fp16, transformer-lens 3.8.0), Cohere's five Tiny Aya models. We rank three-consecutive-layer windows within layers 1 to 12 by their summed absolute MLP attribution.

**What we measured.** With 64 prompts and with 160 prompts (the second an exploratory repeat), the top window changed on 3 of the five models: Fire (10-12 to 2-4), Water (2-4 to 10-12), Base (3-5 to 6-8). It stayed the same on Global (3-5), Earth (3-5). Averaged over the five models, the top window was layers 3 to 5 with both prompt counts; its lead over the next window grew from 0.3 to 1.4 points of attribution share.

In a pre-registered comparison, editing each model's own top window beat the default window in 6 of 10 model-and-seed pairs (p = 0.377), so that hypothesis failed.

**What a maintainer could change.** Report how stable a layer ranking is (for example a bootstrap over prompts, or the margin between the top windows), so a user can see when the top pick is within prompt-sampling noise.

Paper: https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf (Section 4.5). Attribution files: https://github.com/itssaharsh/relapse/tree/main/results/place/circuitkit and https://github.com/itssaharsh/relapse/tree/main/results/place/circuitkit_more

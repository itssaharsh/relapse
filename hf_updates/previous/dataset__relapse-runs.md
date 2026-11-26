---
license: mit
tags: [unlearning, quantization, wmdp, mmlu, auditkit, circuitkit, safetune, tiny-aya, relapse]
pretty_name: Relapse runs, per-question results and logs
---

# Relapse: runs, per-question results and logs

<!-- relapse-status:start -->
> **Status on 27 November 2026.** The `stage2/` folders hold the runs the paper reports, from six pre-registered experiments (the fifth, placement against the tokenizer, is `stage2/shift2/`; the sixth, single-line prompts, `stage2/single_line/`), and `stage2/survey/` holds the survey of public copies, the official repositories' history and the start time and commit of every job. Everything outside `stage2/` and `correct_tokenizer/` is Stage 1, made through the outdated tokenizer setting, and is superseded.
>
> **What the final paper found** ([An Outdated Tokenizer Setting Overstated Unlearning Applied at Layers 5–7 in 16 of 16 Runs, but Not at Layers 3–5](https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf)). Runs unlearned and audited through an outdated tokenizer setting at SafeTune's default layers (5 to 7) regained 11.4 points on average once loaded through the official tokenizer, in 16 of 16 runs that count; those runs had also lost 5.8 MMLU points where they were audited and 1.7 once deployed, and with single-line prompts none of them relapsed. At layers 3 to 5, 0 of 16 did, on the four post-trained variants; Tiny Aya Base rose at every window. In 4-bit files the forgetting held in 107 of 109 runs. Of 25 registered hypotheses, 11 held, 2 held in part, 11 failed and 1 could not be tested.
>
> **Base model and tokenizer revision.** Every Stage 2 job downloaded the main branch of `CohereLabs/tiny-aya-<variant>` when it started; by the job start times and the repositories' history that was the revision carrying Cohere's tokenizer fix (Global `af89d219b5`, Fire `73500c80fc`, Earth `0624fafbe2`, Water `41a99216d1`, Base `d62ac17e24`, all committed 30 September 2026). Stage 1 used the copy Kaggle Models held, which carries the outdated setting.
>
> **Tokenizer checker.** [`scripts/check_tokenizer.py`](https://github.com/itssaharsh/relapse/blob/main/scripts/check_tokenizer.py), added with the paper, tells you whether a copy of Tiny Aya carries the outdated tokenizer setting and what to do about it.
<!-- relapse-status:end -->

> **Correction (Nov 21, 2026).** These runs were made on Kaggle with an outdated tokenizer setting that splits text differently from the way Tiny Aya was trained. Scores here are **superseded**. Everything under `raw/` from Stage 1 was measured that way, and the claims about a relapse at layers 10 to 12 and a repair are withdrawn. [What changed](https://github.com/itssaharsh/relapse#correction-november-21-2026).

## Stage 2 (current): `stage2/`

Six pre-registered experiments on all five Tiny Aya models, scored through the **official tokenizer**; the two tokenizer experiments also unlearn and score through the outdated setting, on purpose. These are the results the [paper and README](https://github.com/itssaharsh/relapse) report.

| Folder | What is in it |
|---|---|
| `stage2/PREREGISTRATION*.md` | The hypotheses, each committed before its runs |
| `stage2/map/` | The 150-run map: fp16, Q4_0 and NF4 scores of every run, right or wrong per question |
| `stage2/deep/` | The relapse follow-up: fresh seeds, real llama.cpp files, bitsandbytes NF4, one part of the model rounded at a time, the repair and its controls |
| `stage2/shift/` | The tokenizer experiment: each run scored through the official and the outdated tokenizer setting |
| `stage2/shift2/` | Placement against the tokenizer (pre-registration 5): layers 3 to 5, 5 to 7 and 7 to 9, unlearned through the outdated setting, scored through both |
| `stage2/single_line/` | Pre-registration 6: the same saved runs scored with single-line prompts through both settings |
| `stage2/survey/` | The survey of public copies, the official repositories' history, the start time and commit of every job, token-level differences; `stage2/check_tokenizer.py` is the checker |
| `stage2/place/`, `stage2/final/` | Where to place the edit, the final models and their full check |

Each folder has a `summary.json` written by the matching `scripts/analyze_*.py` in the repository. The files hold scores only, no question text.

## Stage 1 (superseded): `raw/`

Everything behind [Relapse](https://huggingface.co/spaces/itssaharsh/relapse): forget the hazard, keep the rest, and check
it in the 4-bit file people download. Cohere's [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global), unlearned
with Lexsi [SafeTune](https://github.com/Lexsi-Labs/SafeTune)'s RMU at five layer windows (30 runs, plus the runs that
repair the one window that relapses), scored in up to seven storage formats.

| Folder | What is in it |
|---|---|
| `raw/grid_*.json` | One row per model and format: WMDP-bio and MMLU accuracy, probability on the right answer, and accuracy on each probe set (`probe:hi`, `probe:ar`, `probe:es`, `probe:sw`, `probe:en_para`). `grid_seed1.json` is the second training seed; `grid_deeper.json` is the layer-10-to-12 control in real llama.cpp Q4_0 and Q4_K_M files |
| `raw/sweep_*.json` | The tuning sweeps: SafeTune's RMU defaults, the retune, the CircuitKIT-layer settings, and NPO. `sweep_rmu_controls_*.json` are the controls: the default layers at a matched budget, and the same edit at layers 3 to 5, 10 to 12 and 13 to 15 |
| `raw/sweep_rmu_fix_*.json`, `raw/grid_deeperfix.json` | The repair: the layer-10-to-12 edit retrained with the model seen as its Q4_0 file (on half the steps, on every step, twice as long, a higher learning rate, a second seed), the ablation that rounds only the edited matrices, and real llama.cpp Q4_0 and Q4_K_M files of the repaired model |
| `raw/variant_*.json` | The three placements (default, CircuitKIT, deeper) on Tiny Aya Fire and Tiny Aya Earth, settings unchanged from Global |
| `raw/locate_relapse.json` | Where the relapse comes from: WMDP-bio with only one part of the model rounded to Q4_0 (the layers before the edit, the edited matrices, the rest of the window, the layers after) |
| `raw/edit_vs_rounding.json` | For every run, the size of the edit against the Q4_0 rounding error in the same matrices |
| `raw/robustness_*.json` | Prompt attacks (five worked examples, an expert persona) and the fine-tuning attack (25 LoRA steps on 64 WMDP-bio questions, and on 64 unrelated MMLU questions as a control), scored on the 1,204 WMDP-bio questions not used as examples |
| `raw/items_*.json` | Per question: the four answer probabilities on WMDP-bio and the probability on the right answer for MMLU, keyed `model\|format\|set` |
| `raw/audit_*.json` | AuditKIT paired comparisons: forgetting fp16 against Q4_K_M (and Q4_0 in `audit_formats_deeper.json`), the general-knowledge price against the original at the same format, and the two headline recipes against each other in `audit_recipes.json` (delta, 95% bootstrap interval, p-value, flips) |
| `audit_runs/` | The full AuditKIT runs (every prediction, log-likelihoods, fingerprint) |
| `raw/circuitkit_layers.json`, `circuitkit/` | CircuitKIT EAP-IG attribution on WMDP-bio: per-layer MLP and attention scores, node scores, faithfulness report |
| `raw/bench_*.json` | llama.cpp CPU and transformers T4 speed measurements used in the cost table |
| `logs/` | Kaggle kernel logs of the unlearning runs, the sweeps, the attacks and the final grids |

Run names map to the write-up's methods: `base` = original, `rmu_default_layers` = RMU at SafeTune's default layers 5 to 7,
`rmu_circuit_layers` = RMU at the CircuitKIT layers 7 to 9, `rmu_circuit_layers_qa` = the same trained through a 4-bit
quantizer, `rmu_L7_strong` = RMU at default layers with 2.5x the learning rate, `*_seed1` = the second seed, `rmu_deeper_*` and `X10-12_*` = the control at layers 10 to 12 that relapses in 4-bit files, `X3-5_*` and `X13-15_*` = the other control windows, `D5-7_*` = the default layers at the CircuitKIT runs' budget, `X10-12_*_file*` and `rmu_deeper_file` = the repair runs (`_file` = model seen as its Q4_0 file on half the steps, `_file_p1` = on every step), `X10-12_lr4.5e-4_qa` = only the edited matrices rounded. `results/methods.json` in the code repository maps every sweep run name to its settings; the edited weights of every run are in [itssaharsh/tiny-aya-global-unlearn-bio-sweep](https://huggingface.co/itssaharsh/tiny-aya-global-unlearn-bio-sweep). Formats: fp16, bf16, Q8_0, Q4_0 and nf4 are
exact weight round trips; Q6_K and Q4_K_M are real llama.cpp files, and so is Q4_0 in `grid_deeper.json` and `grid_deeperfix.json`.

Rebuild every number on the Space: `python scripts/archive/stage1/build_results.py --fetch` in [github.com/itssaharsh/relapse](https://github.com/itssaharsh/relapse).

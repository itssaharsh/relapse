---
license: cc-by-nc-4.0
tags: [unlearning, quantization, wmdp, mmlu, safetune, circuitkit, auditkit, tiny-aya, relapse]
pretty_name: Relapse Stage 2 working files (edited matrices and raw job outputs)
---

# Relapse: Stage 2 working files

This is the dataset the Stage 2 jobs of the paper wrote into as they ran: the edited weight matrices of every run and the raw outputs of every job. It is kept so that a scoring job can be rerun without retraining. For reading results, use [`itssaharsh/relapse-runs`](https://huggingface.co/datasets/itssaharsh/relapse-runs), which holds the same per-question results in a tidier layout, or the `results/` folder of the [code repository](https://github.com/itssaharsh/relapse).

**Paper:** [An Outdated Tokenizer Setting Overstated Unlearning Applied at Layers 5–7 in 16 of 16 Runs, but Not at Layers 3–5](https://github.com/itssaharsh/relapse/blob/main/paper/main.pdf). **Code, jobs and pre-registrations:** [github.com/itssaharsh/relapse](https://github.com/itssaharsh/relapse). Status on 26 November 2026 (UTC): made public on 26 November 2026 (UTC); unchanged since the jobs wrote it, apart from this card and the `stage1/` folder.

## What is in it

651 files: 91.3 GB of edited matrices in 58 `.pt` files and 539 MB of other files (JSON, and one binary in `stage1/`).

| Folder | What it holds | Read by | Files | Matrices | Other files |
|---|---|---|---|---|---|
| `map/` | Pre-registration 1, the relapse map: per run `results.json`, `items.json` (one 0/1 per question) and the edited matrices `changed_map.pt`. | `scripts/analyze_map.py`; the matrices by the scoring jobs built on `hf_jobs/header_real.py` (real files, locate) through `PT_REPO` and `PT_PATH`. | 33 | 40.9 GB | 3.4 MB |
| `deep/` | Pre-registration 4, the Base relapse follow-up: reruns, fresh seeds 41 to 48, the repair arms and their controls, with `changed_map.pt`. | `scripts/analyze_deep.py`; the matrices by the real-file, locate, bitsandbytes and runtime jobs. | 65 | 22.2 GB | 2.3 MB |
| `shift/` | Pre-registration 2, the first tokenizer experiment: every run scored through the official and the outdated setting. | `scripts/analyze_shift.py` (from the copies in the code repository's `results/shift/`). | 23 | 0.5 GB | 5.0 MB |
| `shift2/` | Pre-registration 5, placement against the tokenizer: layers 3 to 5, 5 to 7 and 7 to 9, with the edited matrices `changed_shift2.pt`. | The job of pre-registration 6 (`hf_jobs/header_single.py`, at a pinned revision of this dataset); `scripts/analyze_shift2.py` from `results/shift2/`. | 30 | 16.2 GB | 2.0 MB |
| `single_line/` | Pre-registration 6: the saved runs of pre-registration 5 scored with single-line prompts through both settings. | `scripts/analyze_single_line.py` from `results/single_line/`. | 10 | 0.0 GB | 1.1 MB |
| `place/` | Pre-registration 3, placement: layers 3 to 5, each model's CircuitKIT window, 7 to 9 and the default 5 to 7 on seeds 31 and 32. | `scripts/analyze_place.py` from `results/place/`. | 20 | 0.0 GB | 3.8 MB |
| `final/` | The final-model runs of pre-registration 3, amendment 1 (seeds 33 and 34), with `changed_map.pt`. | `scripts/analyze_final.py`; the full check, real-file and publishing jobs. | 15 | 5.4 GB | 0.5 MB |
| `real/` | Outputs of the scoring jobs: real llama.cpp files, locate (one part rounded at a time), the full check, the bitsandbytes loader, the runtime check, publishing, the tokenizer control and the Stage 1 models rescored. Includes AuditKIT run records (`audit_*.json`). | `scripts/analyze_deep.py`, `scripts/analyze_certify.py`, `scripts/rescore_stats.py`. | 390 | 0.0 GB | 512.4 MB |
| `circuitkit/` | CircuitKIT EAP-IG attribution on the five models, 64 prompts: layer importance, node scores, faithfulness report, tokenizer check. | The placement job, to pick each model's window; `scripts/paper_numbers.py` from `results/place/circuitkit/`. | 20 | 0.0 GB | 0.1 MB |
| `circuitkit_more/` | The same with 160 prompts (exploratory). | `scripts/paper_numbers.py` from `results/place/circuitkit_more/`. | 20 | 0.0 GB | 0.1 MB |
| `cons/` | The consolidation pilot (exploratory). | Nothing registered. | 2 | 0.0 GB | 0.1 MB |
| `npo_pilot/` | The NPO with LoRA pilot (exploratory), with `npo_deltas.pt`. | `scripts/paper_numbers.py` counts its runs. | 15 | 2.1 GB | 0.5 MB |
| `smoke/` | A three-batch pipeline test. | Nothing. | 2 | 0.0 GB | 0.0 MB |
| `stage1/` | Six files of the superseded Stage 1 that two job headers read: the edited matrices of five Stage 1 runs (`dl/*/changed_*.pt`) and a `llama-quantize` built from llama.cpp on Kaggle. Copied here on 26 November 2026 (UTC) from a private backup, with unchanged content hashes, so that no job reads a private repository. | `hf_jobs/tok_head.py` (the tokenizer control) and `hf_jobs/header_real.py` (the Stage 1 models rescored; real-file jobs run without `BUILD_QUANTIZE=1`). | 6 | 4.1 GB | 8.2 MB |

A `.pt` file is a PyTorch dictionary: run name to that run's edited `mlp.down_proj` matrices, by parameter name, which the jobs write into Cohere's original model before scoring. Run names give the model, the window, the learning rate, the tokenizer setting where it applies, and the seed, for example `global_W3-5_lr3e-05_outdated_s51`. The jobs are in [`hf_jobs/`](https://github.com/itssaharsh/relapse/tree/main/hf_jobs) and [`kaggle/`](https://github.com/itssaharsh/relapse/tree/main/kaggle); each pre-registration names the settings of its jobs, and `results/survey/jobs.json` in the code repository gives the start time and commit of every job.

## What the text files contain

Scores, per-question 0/1 lists, answer letters and probabilities, attribution scores and job settings. The AuditKIT run records in `real/` also hold the prompt of every question they scored: the public WMDP-bio test questions of [`cais/wmdp`](https://huggingface.co/datasets/cais/wmdp) and the sampled questions of [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu), both MIT-licensed, with the model's answer letter. Nothing here contains text from the gated forget corpus or free-form model output. Paths such as `/tmp/kg/input/...` in these records are directories inside the job's container.

## License and access

CC-BY-NC-4.0, the license of the original Tiny Aya models, from which the edited matrices are derived. Use is also subject to Cohere's terms for Tiny Aya. The benchmark questions quoted in the run records keep their own MIT license, and so does `stage1/dl/final_a/bin/llama-quantize`, a build of llama.cpp. Not gated, like the released models. The matrices are not models on their own: loading them needs the original weights from [`CohereLabs/tiny-aya-<variant>`](https://huggingface.co/CohereLabs/tiny-aya-global), which Cohere gates.

These are research artifacts, not safety guarantees: the paper shows that a short fine-tune restores the unlearned knowledge.

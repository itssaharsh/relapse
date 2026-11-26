---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
tags: [unlearning, safety, rmu, safetune, wmdp, quantization, tiny-aya, lexsi, relapse]
datasets: [cais/wmdp, cais/mmlu]
---

# Relapse: every unlearning run of the sweep

> **Correction (Nov 21, 2026).** These runs were made on Kaggle with an outdated tokenizer setting that splits text differently from the way Tiny Aya was trained. Scores here are **superseded**. The weights in this repo were all trained that way, and the claims here about layers 10 to 12 relapsing and being repaired are withdrawn. [What changed](https://github.com/itssaharsh/relapse#correction-november-21-2026).

All 38 [SafeTune](https://github.com/Lexsi-Labs/SafeTune) RMU runs behind
[Relapse](https://huggingface.co/spaces/itssaharsh/relapse), one small file each. RMU edits only three MLP down-projection
matrices, so a file holds those three tensors in fp16 (the scores below were computed from the fp32 values they were
rounded from). The runs put the same edit at five layer windows: SafeTune's default (5 to 7), the one
[CircuitKIT](https://github.com/Lexsi-Labs/CircuitKIT)'s attribution ranked first (7 to 9), and three controls (3 to 5,
10 to 12, 13 to 15).

**The plain `layers-10-12_*` files are the relapse.** They look unlearned at fp16 and give part of the hazard score back in the
Q4_0 column. They are here as test cases for an unlearning audit, not as models to use. The `layers-10-12_*_file*` files are
the repair attempts: the same edit trained with the whole model seen as its Q4_0 file (weights rounded to 4 bits during training), on half the steps or on every step.
On every step, the fp16 and Q4_0 columns agree in four of five runs; the NF4 column, a format training never showed, does not.

Apply a file to [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) with
`python scripts/patch_checkpoint.py --base <tiny-aya-global dir> --delta <file>.safetensors --out <dir>` from
[github.com/itssaharsh/relapse](https://github.com/itssaharsh/relapse).

Scores are WMDP-bio (forget, lower is better; guessing is 25%) / MMLU (keep, higher is better). The original scores
63.3% / 55.4% at fp16.

| File | Layers | Learning rate | Retain weight | Steering multiplier | Training saw 4-bit rounding | Batches | Seed (0 first, 1 second) | fp16 | Q4_0 | NF4 |
|---|---|---|---|---|---|---|---|---|---|---|
| `default-layers_lr2e-5.safetensors` | 5-7 | 2e-05 | 100 | 3 | no | 250 | 0 | 51.1% / 52.8% | 47.6% / 52.0% | 49.5% / 50.1% |
| `default-layers_lr4e-5.safetensors` | 5-7 | 4e-05 | 100 | 3 | no | 250 | 0 | 38.7% / 51.6% | 36.8% / 50.6% | 37.2% / 46.3% |
| `default-layers_lr7e-5.safetensors` | 5-7 | 7e-05 | 100 | 3 | no | 250 | 0 | 31.3% / 47.3% | 29.9% / 43.6% | 28.6% / 40.3% |
| `default-layers_lr4e-5_4bit-aware.safetensors` | 5-7 | 4e-05 | 100 | 3 | edited weights, half the steps | 250 | 0 | 38.9% / 51.1% | 37.3% / 50.5% | 37.2% / 47.1% |
| `default-layers_lr1e-4.safetensors` | 5-7 | 0.0001 | 100 | 3 | no | 300 | 0 | 25.4% / 43.5% | 26.8% / 41.4% | 26.7% / 36.2% |
| `default-layers_lr3e-4.safetensors` | 5-7 | 0.0003 | 100 | 3 | no | 300 | 0 | 28.1% / 45.7% | not run | not run |
| `default-layers_a10_lr1e-4.safetensors` | 5-7 | 0.0001 | 10 | 3 | no | 300 | 0 | 26.0% / 35.5% | not run | not run |
| `default-layers_lr4e-5_seed1.safetensors` | 5-7 | 4e-05 | 100 | 3 | no | 250 | 1 | 44.4% / 51.7% | 42.3% / 52.5% | 45.2% / 49.7% |
| `circuitkit-layers_lr1e-4.safetensors` | 7-9 | 0.0001 | 100 | 3 | no | 300 | 0 | 44.8% / 54.1% | 42.6% / 52.3% | 43.0% / 50.9% |
| `circuitkit-layers_a300_lr2e-4.safetensors` | 7-9 | 0.0002 | 300 | 3 | no | 300 | 0 | 44.9% / 54.8% | 42.6% / 54.5% | 43.8% / 52.4% |
| `circuitkit-layers_lr1.5e-4.safetensors` | 7-9 | 0.00015 | 100 | 3 | no | 300 | 0 | 36.8% / 53.2% | 35.9% / 52.7% | 35.3% / 51.2% |
| `circuitkit-layers_lr1.5e-4_4bit-aware.safetensors` | 7-9 | 0.00015 | 100 | 3 | edited weights, half the steps | 300 | 0 | 34.0% / 54.5% | 35.6% / 52.9% | 34.7% / 50.8% |
| `circuitkit-layers_lr2e-4.safetensors` | 7-9 | 0.0002 | 100 | 3 | no | 300 | 0 | 33.6% / 53.6% | 34.2% / 52.9% | 33.1% / 51.0% |
| `circuitkit-layers_lr2e-4_4bit-aware.safetensors` | 7-9 | 0.0002 | 100 | 3 | edited weights, half the steps | 300 | 0 | 32.4% / 54.5% | 33.1% / 53.0% | 31.5% / 51.3% |
| `circuitkit-layers_lr1.5e-4_m5_4bit-aware.safetensors` | 7-9 | 0.00015 | 100 | 5 | edited weights, half the steps | 300 | 0 | 31.0% / 52.7% | 31.3% / 52.0% | 30.1% / 49.4% |
| `circuitkit-layers_lr1.5e-4_seed1.safetensors` | 7-9 | 0.00015 | 100 | 3 | no | 300 | 1 | 45.6% / 55.7% | 43.8% / 54.8% | 44.1% / 52.4% |
| `default-layers_lr4e-5_nb300.safetensors` | 5-7 | 4e-05 | 100 | 3 | no | 300 | 0 | 35.8% / 50.4% | 34.2% / 48.6% | 33.6% / 45.2% |
| `default-layers_lr7e-5_nb300.safetensors` | 5-7 | 7e-05 | 100 | 3 | no | 300 | 0 | 29.9% / 47.3% | 29.9% / 43.6% | 28.4% / 40.7% |
| `default-layers_lr4e-5_m5.safetensors` | 5-7 | 4e-05 | 100 | 5 | no | 300 | 0 | 27.4% / 44.4% | 26.1% / 42.0% | 26.2% / 38.3% |
| `default-layers_a300_lr1e-4.safetensors` | 5-7 | 0.0001 | 300 | 3 | no | 300 | 0 | 37.2% / 51.6% | 35.7% / 50.4% | 35.7% / 48.8% |
| `circuitkit-layers_lr1.5e-4_rep.safetensors` | 7-9 | 0.00015 | 100 | 3 | no | 300 | 0 | 36.3% / 53.2% | 35.7% / 53.3% | 35.5% / 50.8% |
| `layers-3-5_lr1.5e-5.safetensors` | 3-5 | 1.5e-05 | 100 | 3 | no | 300 | 0 | 45.8% / 54.5% | 44.2% / 53.1% | 44.9% / 52.0% |
| `layers-3-5_lr3e-5.safetensors` | 3-5 | 3e-05 | 100 | 3 | no | 300 | 0 | 38.6% / 53.8% | 37.2% / 52.6% | 37.2% / 51.6% |
| `layers-3-5_lr6e-5.safetensors` | 3-5 | 6e-05 | 100 | 3 | no | 300 | 0 | 34.4% / 54.1% | 32.0% / 52.9% | 32.5% / 51.2% |
| `layers-10-12_lr2e-4.safetensors` | 10-12 | 0.0002 | 100 | 3 | no | 300 | 0 | 45.2% / 53.1% | 50.2% / 53.2% | 48.4% / 49.5% |
| `layers-10-12_lr4.5e-4.safetensors` | 10-12 | 0.00045 | 100 | 3 | no | 300 | 0 | 41.6% / 53.9% | 49.3% / 53.3% | 42.2% / 50.6% |
| `layers-10-12_lr1e-3.safetensors` | 10-12 | 0.001 | 100 | 3 | no | 300 | 0 | 28.9% / 50.6% | 31.0% / 51.0% | 29.1% / 50.2% |
| `layers-10-12_lr4.5e-4_seed1.safetensors` | 10-12 | 0.00045 | 100 | 3 | no | 300 | 1 | 50.9% / 51.7% | 55.5% / 52.5% | 53.0% / 51.3% |
| `layers-13-15_lr8e-4.safetensors` | 13-15 | 0.0008 | 100 | 3 | no | 300 | 0 | 58.2% / 55.0% | 57.7% / 54.2% | 54.0% / 51.5% |
| `layers-13-15_lr2e-3.safetensors` | 13-15 | 0.002 | 100 | 3 | no | 300 | 0 | 46.3% / 51.6% | 43.8% / 52.6% | 45.2% / 50.7% |
| `layers-10-12_lr4.5e-4_4bit-aware.safetensors` | 10-12 | 0.00045 | 100 | 3 | edited weights, half the steps | 300 | 0 | 32.1% / 51.0% | 38.6% / 51.8% | 30.2% / 50.0% |
| `layers-10-12_lr4.5e-4_file-half.safetensors` | 10-12 | 0.00045 | 100 | 3 | whole model as its Q4_0 file, half the steps | 300 | 0 | 37.5% / 51.8% | 40.9% / 51.3% | 37.5% / 50.2% |
| `layers-10-12_lr4.5e-4_file-half_nb600.safetensors` | 10-12 | 0.00045 | 100 | 3 | whole model as its Q4_0 file, half the steps | 600 | 0 | 28.7% / 49.6% | 32.0% / 48.3% | 27.8% / 47.3% |
| `layers-10-12_lr4.5e-4_file.safetensors` | 10-12 | 0.00045 | 100 | 3 | whole model as its Q4_0 file, every step | 300 | 0 | 49.1% / 52.8% | 49.1% / 53.0% | 47.9% / 50.0% |
| `layers-10-12_lr4.5e-4_file_nb600.safetensors` | 10-12 | 0.00045 | 100 | 3 | whole model as its Q4_0 file, every step | 600 | 0 | 41.5% / 50.6% | 41.0% / 48.9% | 35.3% / 47.2% |
| `layers-10-12_lr1e-3_file.safetensors` | 10-12 | 0.001 | 100 | 3 | whole model as its Q4_0 file, every step | 300 | 0 | 26.1% / 44.2% | 27.0% / 43.8% | 26.7% / 42.3% |
| `layers-10-12_lr4.5e-4_file_seed1.safetensors` | 10-12 | 0.00045 | 100 | 3 | whole model as its Q4_0 file, every step | 300 | 1 | 49.6% / 52.5% | 52.1% / 52.4% | 51.9% / 51.6% |
| `layers-10-12_lr4.5e-4_file_nb600_seed1.safetensors` | 10-12 | 0.00045 | 100 | 3 | whole model as its Q4_0 file, every step | 600 | 1 | 32.5% / 45.1% | 30.0% / 41.3% | 39.7% / 46.0% |

Four of these are also published as full checkpoints with GGUF files in the
[Relapse models collection](https://huggingface.co/collections/itssaharsh/relapse-unlearned-tiny-aya-models). All scores,
per-question results and logs are in [itssaharsh/relapse-runs](https://huggingface.co/datasets/itssaharsh/relapse-runs).

A research artifact. Outside layers 10 to 12 the forgetting holds across file formats; for the headline models it also
holds across languages and prompt attacks. It does not survive fine-tuning on the topic. The base model's license and the Cohere Labs acceptable use policy apply (non-commercial).

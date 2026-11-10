# Kaggle runs

Every number in Relapse comes from these scripts, run as Kaggle kernels on 2x T4 (16 GB each, no bf16) with internet on.
Each kernel is `common.py` followed by one numbered script (`./make_kernel.sh 05_rmu_default_layers_grid.py > run.py`); `00_baseline_wmdp.py` and `02_circuitkit_wmdp_layers.py` are standalone.

| Script | What it produces | Attach |
|---|---|---|
| `00_baseline_wmdp.py` | Tiny Aya Global on WMDP-bio/cyber/chem and MMLU at fp16 and NF4 | model |
| `01_gguf_roundtrip_check.py` | proves the llama.cpp path: HF -> GGUF f16 -> back is exact; Q8_0 and Q4_K_M load | model (CPU kernel) |
| `02_circuitkit_wmdp_layers.py` | CircuitKIT EAP-IG attribution on WMDP-bio: per-layer MLP and attention scores | model |
| `03_probe_translations.py` | WMDP-bio questions in Hindi, Arabic, Spanish, Swahili and reworded English, back-translation scored | model |
| `04_rmu_retune.py` | SafeTune RMU sweep (layer, alpha, learning rate) with loss logs | model, forget corpus |
| `05_rmu_default_layers_grid.py` | RMU at the default layers 5-7 (three strengths + quantization-aware) across fp16, Q8_0, Q4_0, NF4 | model, forget corpus |
| `06_rmu_circuitkit_layers_grid.py` | RMU at the CircuitKIT layers 7-9 (+ quantization-aware) across fp16, Q4_0, NF4 and probes | model, forget corpus, probes |
| `07_npo_lora_sweep.py` | SafeTune NPO (LoRA) sweep, kept for the record: not selective on Tiny Aya | model, forget corpus |
| `08_final_grid.py` | the published grid: every model x fp16, bf16, Q8_0, Q4_0, NF4 (exact round trips) and Q6_K, Q4_K_M (real llama.cpp), probes, per-question probabilities, AuditKIT paired comparisons, Q4_K_M GGUF files | model, tools, probes, the unlearning kernel's output |

Attachments: model = Kaggle Models `cohere-labs/tiny-aya/transformers/global/1`. Forget corpus = the WMDP bio forget corpus
(`cais/wmdp-bio-forget-corpus`, gated by CAIS; request access yourself, then put the `text` field into `bio_forget.jsonl`
in a private Kaggle dataset; it is never redistributed here). Tools = a static `llama-quantize` binary (the scripts build it
when it is missing). Probes = `probes.json` from the `relapse-wmdp-bio-probes` dataset on Hugging Face.
Environment variables at the top of a kernel select the runs (`RUNS`, `TAG`, `CONFIGS`, `PRECS`); the defaults are the published ones.

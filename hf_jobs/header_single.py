# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch", "transformers==5.20.2", "accelerate", "datasets", "huggingface_hub", "numpy", "safetensors", "sentencepiece", "protobuf", "tqdm", "psutil"]
# ///
# Pre-registration 6 on Hugging Face Jobs: rescore the saved runs of pre-registration 5 with single-line prompts. Scoring only, nothing is trained.
# Built by hf_jobs/build_single_line.sh from this header, kaggle/common.py and kaggle/20_single_line.py. Every revision is pinned by commit hash.
import atexit, json as _j, os as _o, time as _t
V = _o.environ["VARIANT"]
MODEL_REV = {"global": "af89d219b53ed9b13b8a4645f9c8028973510324", "fire": "73500c80fcb6c8bea3e830c21d5d2920249a0e1a", "earth": "0624fafbe200f930b89566fc4ec4da314ba48d89", "water": "41a99216d12b3ef4881df762934186e5e4d2bbc1", "base": "d62ac17e241c871fc52e9a95d7b0e2615189d7d5"}[V]      # model and tokenizer files: the revision pre-registration 5 used (Cohere's fix of 30 September 2026)
WEIGHTS_REV = "08a860a2bdc5a49581f01ce58e28bb78120872a7"      # itssaharsh/relapse-stage2: the edited matrices saved by the jobs of pre-registration 5
WMDP_REV, MMLU_REV = "7125571f22f032c56415e7980f48d877dd830ff8", "c30699e8356da336a370243923dbaf21066bb9fe"      # cais/wmdp and cais/mmlu
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
OUT_REPO, OUT_DIR = "itssaharsh/relapse-stage2", "/tmp/kg/working"
_o.makedirs(OUT_DIR, exist_ok=True)
snapshot_download(f"CohereLabs/tiny-aya-{V}", revision=MODEL_REV, local_dir=f"/tmp/kg/input/models/cohere-labs/tiny-aya/transformers/{V}/1",
                  allow_patterns=["*.safetensors", "*.json", "tokenizer*", "special_tokens*"])
PT = {s: hf_hub_download(OUT_REPO, f"shift2/{V}_seed{s}/changed_shift2.pt", repo_type="dataset", revision=WEIGHTS_REV) for s in (51, 52)}
_api = HfApi()
def _push(names=("results.json", "items.json"), tries=15):
    for n in names:
        p = f"{OUT_DIR}/{n}"
        if _o.path.exists(p):
            for i in range(tries):
                try:
                    _api.upload_file(path_or_fileobj=p, path_in_repo=f"single_line/{V}/{n}", repo_id=OUT_REPO, repo_type="dataset", commit_message=f"single_line {V}: {n}"); break
                except Exception as e:
                    print("upload retry", n, repr(e)[:160], flush=True); _t.sleep(min(60 * (i + 1), 300))
atexit.register(_push)

# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch", "transformers==5.20.2", "accelerate", "datasets", "huggingface_hub", "bitsandbytes>=0.45",
#                 "safetune==0.1.13", "numpy", "safetensors", "sentencepiece", "protobuf", "pyyaml", "tqdm", "psutil"]
# ///
# Relapse map on Hugging Face Jobs: one Tiny Aya variant, one seed, every layer window at three strengths.
# Built by hf_jobs/build.sh from this header, kaggle/common.py and kaggle/06_rmu_circuitkit_layers_grid.py.
# Scores and edited weights go to a private repo. The gated forget corpus stays inside the job and is never uploaded.
import atexit, io, json as _j, os as _o, shutil as _sh, tarfile, threading, time as _t, urllib.request
V, S = _o.environ["VARIANT"], int(_o.environ["SEED"])
RUN = f"{V}_seed{S}" + _o.environ.get("RUN_SUFFIX", "")
LADDER = {(3, 5): (1.5e-5, 3e-5, 6e-5), (5, 7): (2e-5, 4e-5, 7e-5), (7, 9): (1e-4, 1.5e-4, 2.5e-4),
          (10, 12): (2e-4, 4.5e-4, 1e-3), (13, 15): (1e-3, 2e-3, 4e-3)}
if "CONFIGS" not in _o.environ:
    _o.environ["CONFIGS"] = _j.dumps([{"name": f"{V}_W{a}-{b}_lr{lr:g}_s{S}", "layer": b, "alpha": 100, "lr": lr, "mult": 3, "seed": S}
                                      for (a, b), lrs in LADDER.items() for lr in lrs])
for k, v in (("FROZEN_FP16", "1"), ("SAVE_ITEMS", "1"), ("OUT", "changed_map.pt"), ("PRECS", "fp16,Q4_0,nf4")):
    _o.environ.setdefault(k, v)
from huggingface_hub import HfApi, snapshot_download
from datasets import load_dataset
OUT_REPO, OUT_DIR = "itssaharsh/relapse-stage2", "/tmp/kg/working"
_o.makedirs(OUT_DIR, exist_ok=True)
snapshot_download(f"CohereLabs/tiny-aya-{V}", local_dir=f"/tmp/kg/input/models/cohere-labs/tiny-aya/transformers/{V}/1",
                  allow_patterns=["*.safetensors", "*.json", "tokenizer*", "special_tokens*"])
_o.makedirs("/tmp/kg/input/forget", exist_ok=True)
with open("/tmp/kg/input/forget/bio_forget.jsonl", "w") as f:
    for r in load_dataset("cais/wmdp-bio-forget-corpus", split="train"):
        f.write(_j.dumps({"text": r["text"]}) + "\n")
_sha = _o.environ["LLAMA_COMMIT"]
with urllib.request.urlopen(f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{_sha}", timeout=300) as r:
    tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz").extractall("/tmp/_lc", filter="data")
_sh.move(f"/tmp/_lc/llama.cpp-{_sha}", "/tmp/llama.cpp")
_api = HfApi()
def _push(names):
    for n in names:
        p = f"{OUT_DIR}/{n}"
        if _o.path.exists(p):
            for i in range(4):
                try:
                    _api.upload_file(path_or_fileobj=p, path_in_repo=f"map/{RUN}/{n}", repo_id=OUT_REPO, repo_type="dataset", commit_message=f"{RUN}: {n}"); break
                except Exception as e:
                    print("upload retry", n, repr(e)[:160], flush=True); _t.sleep(20)
def _loop():
    while True:
        _t.sleep(900); _push(["results.json", "items.json"])
threading.Thread(target=_loop, daemon=True).start()
atexit.register(lambda: _push(["results.json", "items.json", "changed_map.pt"]))

# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch", "transformers==5.20.2", "accelerate", "datasets", "huggingface_hub", "bitsandbytes>=0.45",
#                 "auditkit==1.1.6", "peft", "numpy", "safetensors", "sentencepiece", "protobuf", "pyyaml", "tqdm", "psutil"]
# ///
# Real llama.cpp files + AuditKIT for saved runs, on Hugging Face Jobs.
# Built by hf_jobs/build_real.sh from this header, kaggle/common.py and kaggle/08_final_grid.py.
import atexit, glob as _g, io, json as _j, os as _o, shutil as _sh, subprocess as _sp, sys as _s, tarfile, urllib.request
V, TAG = _o.environ["VARIANT"], _o.environ["TAG"]
_o.environ["PATH"] = _o.path.dirname(_s.executable) + ":" + _o.environ.get("PATH", "")
_BUILD = _o.environ.get("BUILD_QUANTIZE") == "1"      # build llama-quantize from the pinned llama.cpp commit instead of using the saved Stage 1 binary
_sp.run("apt-get update -qq && apt-get install -y -qq libgomp1 libstdc++6" + (" cmake build-essential" if _BUILD else ""), shell=True, check=False)
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
OUT_REPO, OUT_DIR = "itssaharsh/relapse-stage2", "/tmp/kg/working"
_o.makedirs(OUT_DIR, exist_ok=True); _o.makedirs("/tmp/kg/input/run", exist_ok=True)
snapshot_download(f"CohereLabs/tiny-aya-{V}", local_dir=f"/tmp/kg/input/models/cohere-labs/tiny-aya/transformers/{V}/1",
                  allow_patterns=["*.safetensors", "*.json", "tokenizer*", "special_tokens*"])
# Stage 1 files (the default when PT_REPO is not set, and the prebuilt llama-quantize): the public copy in the stage1/ folder of the dataset itssaharsh/relapse-stage2.
# The jobs of 21 November 2026 read the same files, by the same paths without the stage1/ prefix, from a private backup; the content hashes are unchanged.
_STAGE1 = "PT_REPO" not in _o.environ
for _p in _o.environ["PT_PATH"].split(","):
    _sh.copy(hf_hub_download(_o.environ.get("PT_REPO", "itssaharsh/relapse-stage2"), ("stage1/" if _STAGE1 else "") + _p, repo_type=_o.environ.get("PT_TYPE", "dataset")), "/tmp/kg/input/run/" + _o.path.basename(_p))
if not _BUILD:
    _sh.copy(hf_hub_download("itssaharsh/relapse-stage2", "stage1/dl/final_a/bin/llama-quantize", repo_type="dataset"), "/tmp/kg/input/run/llama-quantize")
_gitsha, _sha = _o.environ["REPO_COMMIT"], _o.environ["LLAMA_COMMIT"]
urllib.request.urlretrieve(f"https://raw.githubusercontent.com/itssaharsh/relapse/{_gitsha}/" + "kaggle" + "/auditkit_mcq.py", "/tmp/kg/input/run/auditkit_mcq.py")
with urllib.request.urlopen(f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{_sha}", timeout=300) as r:
    tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz").extractall("/tmp/_lc", filter="data")
_sh.move(f"/tmp/_lc/llama.cpp-{_sha}", "/tmp/llama.cpp")
_bp = "/tmp/llama.cpp/conversion/base.py"; _src = open(_bp).read()      # same tokenizer-hash patch as setup_llamacpp
open(_bp, "w").write(_src.replace('raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")',
                                  'res = "tiny_aya"; logger.warning("unknown chkhsh %s -> tiny_aya", chkhsh)'))
def _push():
    import time
    for i in range(15):      # the hub limits commits per hour; wait longer each time
        try:
            HfApi().upload_folder(repo_id=OUT_REPO, repo_type="dataset", folder_path=OUT_DIR, path_in_repo=f"real/{TAG}", allow_patterns=["*.json"], commit_message=f"real files: {TAG}"); return
        except Exception as e:
            print("upload retry", repr(e)[:160], flush=True); time.sleep(min(60 * (i + 1), 300))
atexit.register(_push)

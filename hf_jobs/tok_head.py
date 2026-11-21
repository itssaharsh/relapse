# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch", "transformers==5.20.2", "accelerate", "datasets", "huggingface_hub", "numpy", "safetensors", "sentencepiece", "protobuf"]
# ///
# Control: the same weights in the same job, scored with the tokenizer settings Kaggle shipped and with the current official ones.
import os as _o, shutil as _sh
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
_o.makedirs("/tmp/kg/working", exist_ok=True); _o.makedirs("/tmp/kg/input/run", exist_ok=True)
snapshot_download("CohereLabs/tiny-aya-global", local_dir="/tmp/kg/input/models/cohere-labs/tiny-aya/transformers/global/1",
                  allow_patterns=["*.safetensors", "*.json", "tokenizer*", "special_tokens*"])
for _p in ("dl/rmu3/changed_rmu3.pt", "dl/rmu4/changed_rmu4.pt", "dl/relapse-ctrl-deeper/changed_ctrl_deeper.pt"):
    _sh.copy(hf_hub_download("itssaharsh/relapse-backup", _p), "/tmp/kg/input/run/" + _o.path.basename(_p))

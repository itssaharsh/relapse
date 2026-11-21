# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch", "transformers==5.20.2", "transformer-lens==3.8.0", "circuitkit==0.2.1", "accelerate", "datasets",
#                 "huggingface_hub>=1.5,<2", "safetensors", "sentencepiece", "protobuf", "psutil"]
# ///
# CircuitKIT EAP-IG on one Tiny Aya variant through the official tokenizer, with the Stage 1 settings
# (kaggle/02_circuitkit_wmdp_layers.py). Attributions go to the private stage-2 repo.
import importlib.util, json, os, re, sys, time, traceback
V = os.environ["VARIANT"]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["CIRCUITKIT_SKIP_MEM_GUARD"] = "1"
from huggingface_hub import HfApi, snapshot_download
MD = f"/tmp/kg/input/models/cohere-labs/tiny-aya/transformers/{V}/1"
OUT = "/tmp/kg/working"; os.makedirs(OUT, exist_ok=True)
snapshot_download(f"CohereLabs/tiny-aya-{V}", local_dir=MD, allow_patterns=["*.safetensors", "*.json", "tokenizer*", "special_tokens*"])
# memory patches, as in Stage 1: avoid a [b,p,h,d_head,d_model] tensor for hook_result and a 1 GiB transposed unembedding copy
TL = importlib.util.find_spec("transformer_lens").submodule_search_locations[0]
_p = f"{TL}/components/abstract_attention.py"; _s = open(_p).read(); _old = "unhooked_result = (z * w).sum(-2)"; assert _old in _s
open(_p, "w").write(_s.replace(_old, 'unhooked_result = torch.einsum("bphd,hdm->bphm", z[..., 0], w[0, 0])'))
_u = f"{TL}/components/unembed.py"; _us = open(_u).read(); assert "self.W_U.T.contiguous()" in _us
open(_u, "w").write(_us.replace("self.W_U.T.contiguous()", "self.W_U.T"))
import circuitkit as ck  # before transformer_lens
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from circuitkit.tasks.builtins import wmdp as _w
from circuitkit.tasks._chat import to_tokens as _tt
MAXLEN, NEX = 80, 64
def agg(scores):
    layers = {l: {"mlp": 0.0, "attn": 0.0, "mlp_abs": 0.0, "attn_abs": 0.0} for l in range(36)}
    for k, v in scores.items():
        v = float(v)
        m = re.match(r"(?:m|MLP )(\d+)$", k); a = re.match(r"(?:a(\d+)\.h(\d+)|A(\d+)\.(\d+))$", k)
        if m:
            L = layers[int(m.group(1))]; L["mlp"] += v; L["mlp_abs"] += abs(v)
        elif a:
            g = [x for x in a.groups() if x is not None]; L = layers[int(g[0])]; L["attn"] += v; L["attn_abs"] += abs(v)
    return layers
def _short_filter(self, exs, model, top_percent=0.1, *, apply=False):
    keep = []
    for ex, cfg in exs:
        try:
            p = self._wrap_wmdp_prompt(model, self._format_wmdp_prompt(ex["question"], list(ex["choices"])), apply=apply)
            if _tt(model, p, templated=apply).size(1) <= MAXLEN: keep.append((ex, cfg))
        except Exception:
            pass
    print("SHORT FILTER kept", len(keep), "of", len(exs), flush=True)
    return keep[:NEX]
_w.WMDPTaskSpec._filter_longest_sequences_global = _short_filter
api = HfApi()
def push():
    for f in ("layer_importance.json", "node_scores.json", "tokenizer_check.json", "faithfulness.json", "error.txt"):
        if os.path.exists(f"{OUT}/{f}"):
            api.upload_file(path_or_fileobj=f"{OUT}/{f}", path_in_repo=f"circuitkit/{V}/{f}", repo_id="itssaharsh/relapse-stage2", repo_type="dataset", commit_message=f"circuitkit {V}: {f}")
t0 = time.time()
try:
    model = ck.load_model(MD, dtype="float16", algorithm="eap-ig")
    official = AutoTokenizer.from_pretrained(MD)
    qs = [r["question"] for r in load_dataset("cais/wmdp", "wmdp-bio", split="test")][:300]
    same = sum(model.tokenizer(q, add_special_tokens=False).input_ids == official(q, add_special_tokens=False).input_ids for q in qs)
    check = {"official_class": type(official).__name__, "circuitkit_class": type(model.tokenizer).__name__, "identical_prompts": same, "of": len(qs)}
    json.dump(check, open(f"{OUT}/tokenizer_check.json", "w"), indent=1); print(check, flush=True)
    assert type(official).__name__ != "CohereTokenizerFast" and same == len(qs), "CircuitKIT is not using the official tokenizer"
    model.unembed.W_U = torch.nn.Parameter(model.embed.W_E.T.detach(), requires_grad=False)  # tied weights: drop the duplicate copy
    torch.cuda.empty_cache()
    circuit = ck.discover(model, "wmdp", algorithm="eap-ig", n_examples=NEX, batch_size=1, ig_steps=5, model_name=MD,
                          samples_per_config=700, configs="wmdp-bio", sparsity=0.5, scope="both", output_path=OUT + "/circuit.pt")
    scores = dict(circuit.scores); layers = agg(scores)
    score = lambda a: sum(layers[l]["mlp_abs"] for l in range(a, a + 3))
    a = max(range(1, 11), key=score)
    res = {"variant": V, "method": "circuitkit eap-ig", "n_examples": NEX, "n_scored_nodes": len(scores), "n_circuit_nodes": len(circuit.nodes),
           "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "tokenizer": check, "window_1_12": f"{a}-{a + 2}", "per_layer": layers}
    json.dump(scores, open(f"{OUT}/node_scores.json", "w")); json.dump(res, open(f"{OUT}/layer_importance.json", "w"), indent=1)
    print("WINDOW", V, res["window_1_12"], "secs", round(time.time() - t0), flush=True)
    try:
        rep = ck.faithfulness(model, circuit, "wmdp", n_examples=32, batch_size=1, configs="wmdp-bio")
        json.dump(rep.to_dict() if hasattr(rep, "to_dict") else str(rep), open(f"{OUT}/faithfulness.json", "w"), indent=1, default=str)
    except Exception:
        traceback.print_exc()
    push()
except Exception:
    traceback.print_exc()
    open(f"{OUT}/error.txt", "w").write(traceback.format_exc()); push(); sys.exit(1)

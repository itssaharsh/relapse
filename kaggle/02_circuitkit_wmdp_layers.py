import subprocess, sys, os, json, re, time, glob, traceback
def sh(c):
    print("$", c, flush=True)
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr)[-3000:], flush=True)
    return r.returncode
sh("pip list 2>/dev/null | grep -iE '^(torch|transformers|transformer-lens|numpy|datasets|huggingface) '")
sh("pip install circuitkit 2>&1 | tail -15")
sh("pip list 2>/dev/null | grep -iE '^(torch|transformers|transformer-lens|circuitkit|numpy|datasets) '")
# memory patch: TL abstract_attention builds a [b,p,h,d_head,d_model] tensor for hook_result; use einsum
_p = glob.glob("/usr/local/lib/python3*/dist-packages/transformer_lens/components/abstract_attention.py")[0]
_s = open(_p).read()
_old = "unhooked_result = (z * w).sum(-2)"
assert _old in _s
open(_p, "w").write(_s.replace(_old, 'unhooked_result = torch.einsum("bphd,hdm->bphm", z[..., 0], w[0, 0])'))
_u = _p.replace("abstract_attention.py", "unembed.py")
_us = open(_u).read(); assert "self.W_U.T.contiguous()" in _us
open(_u, "w").write(_us.replace("self.W_U.T.contiguous()", "self.W_U.T"))  # avoid 1GiB transposed copy
os.environ["CIRCUITKIT_SKIP_MEM_GUARD"] = "1"
OUT = "/kaggle/working"
cands = glob.glob("/kaggle/input/**/config.json", recursive=True)
print(cands)
MD = os.path.dirname([c for c in cands if "tiny-aya" in c][0])
print("MODEL_DIR", MD, os.listdir(MD), flush=True)
import circuitkit as ck  # before transformer_lens
import torch
res = {"model_dir": MD, "versions": {"torch": torch.__version__}}
def agg(scores):
    layers = {l: {"mlp": 0.0, "attn": 0.0, "mlp_abs": 0.0, "attn_abs": 0.0, "heads": {}} for l in range(36)}
    for k, v in scores.items():
        v = float(v)
        m = re.match(r"(?:m|MLP )(\d+)$", k)
        a = re.match(r"(?:a(\d+)\.h(\d+)|A(\d+)\.(\d+))$", k)
        if m:
            L = layers[int(m.group(1))]; L["mlp"] += v; L["mlp_abs"] += abs(v)
        elif a:
            g = [x for x in a.groups() if x is not None]; L = layers[int(g[0])]
            L["attn"] += v; L["attn_abs"] += abs(v); L["heads"][g[1]] = v
    return layers
from circuitkit.tasks.builtins import wmdp as _w
from circuitkit.tasks._chat import to_tokens as _tt
MAXLEN = int(os.environ.get("MAXLEN", 80)); NEX = int(os.environ.get("NEX", 64))
def _short_filter(self, exs, model, top_percent=0.1, *, apply=False):
    keep = []
    for ex, cfg in exs:
        try:
            p = self._wrap_wmdp_prompt(model, self._format_wmdp_prompt(ex["question"], list(ex["choices"])), apply=apply)
            if _tt(model, p, templated=apply).size(1) <= MAXLEN:
                keep.append((ex, cfg))
        except Exception:
            pass
    print("SHORT FILTER kept", len(keep), "of", len(exs), flush=True)
    return keep[:NEX]
_w.WMDPTaskSpec._filter_longest_sequences_global = _short_filter
t0 = time.time()
try:
    model = ck.load_model(MD, dtype="float16", algorithm="eap-ig")
    try:
        model.unembed.W_U = torch.nn.Parameter(model.embed.W_E.T.detach(), requires_grad=False)  # tied: drop duplicate 1GiB copy
    except Exception as e:
        print("tie failed", e)
    torch.cuda.empty_cache()
    print("loaded", time.time() - t0, torch.cuda.memory_allocated() / 1e9, flush=True)
    nex = NEX
    circuit = ck.discover(model, "wmdp", algorithm="eap-ig", n_examples=nex, batch_size=1,
                          ig_steps=5, model_name=MD, samples_per_config=700, configs="wmdp-bio", sparsity=0.5, scope="both",
                          output_path=OUT + "/circuit.pt")
    print("discover done", time.time() - t0, "peak GB", torch.cuda.max_memory_allocated() / 1e9, flush=True)
    scores = dict(circuit.scores)
    layers = agg(scores)
    res.update(method="circuitkit eap-ig", n_examples=nex, n_scored_nodes=len(scores),
               n_circuit_nodes=len(circuit.nodes), peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    res["per_layer"] = layers
    res["rank_mlp_abs"] = sorted(layers, key=lambda l: -layers[l]["mlp_abs"])
    res["rank_attn_abs"] = sorted(layers, key=lambda l: -layers[l]["attn_abs"])
    res["rank_combined_abs"] = sorted(layers, key=lambda l: -(layers[l]["mlp_abs"] + layers[l]["attn_abs"]))
    json.dump(scores, open(OUT + "/node_scores.json", "w"))
    json.dump(res, open(OUT + "/layer_importance.json", "w"), indent=1)
    try:
        rep = ck.faithfulness(model, circuit, "wmdp", n_examples=32, batch_size=1, configs="wmdp-bio")
        d = rep.to_dict() if hasattr(rep, "to_dict") else str(rep)
        json.dump(d, open(OUT + "/faithfulness.json", "w"), indent=1, default=str)
        print("FAITH", rep, flush=True)
    except Exception:
        traceback.print_exc()
        open(OUT + "/faithfulness_error.txt", "w").write(traceback.format_exc())
except Exception:
    traceback.print_exc()
    open(OUT + "/circuitkit_error.txt", "w").write(traceback.format_exc())
    sys.exit(1)
print("TOP10 mlp", [(l, round(res["per_layer"][l]["mlp_abs"], 4)) for l in res["rank_mlp_abs"][:10]])

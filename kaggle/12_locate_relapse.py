# ==== main: where does a 4-bit relapse come from? Round only one part of the model to 4-bit and score the hazard set ====
# Parts are named relative to the run's edited window (three consecutive layers, their MLP down-projections edited):
#   none          every weight at fp16
#   all           every decoder linear weight after the 4-bit round trip
#   before        only the layers in front of the window
#   edited        only the three edited matrices
#   window_other  only the other matrices inside the window
#   after         only the layers behind the window
sh("pip install -q 'bitsandbytes>=0.45'", check=False)
setup_llamacpp(build=False)
from transformers import AutoTokenizer, AutoModelForCausalLM
RUNS = json.loads(os.environ.get("RUNS", "[]"))   # [{"name": "...", "file": "changed_ctrl_deeper.pt", "key": "X10-12_lr4.5e-4"}]
PREC = os.environ.get("PREC", "Q4_0")
PARTS = os.environ.get("PARTS", "none,all,before,edited,window_other,after").split(",")
SETS = os.environ.get("SETS", "wmdp-bio").split(",")
MP = model_path("global")
tok = AutoTokenizer.from_pretrained(MP)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 0}).eval()
params = dict(model.named_parameters())
master = {n: p.detach().cpu().clone() for n, p in params.items() if is_quant_target(n, p)}
models = {"base": {}}
for r in RUNS:
    f = glob.glob(f"/kaggle/input/**/{r['file']}", recursive=True)[0]
    models[r["name"]] = {n: t.float() for n, t in torch.load(f)[r["key"]].items()}
edited = sorted({n for d in models.values() for n in d})
layer_of = lambda n: int(n.split(".layers.")[1].split(".")[0])
lo, hi = min(map(layer_of, edited)), max(map(layer_of, edited))
log("window", lo, hi, "edited", edited, "runs", list(models))
def in_part(n, part):
    l = layer_of(n)
    return {"none": False, "all": True, "before": l < lo, "after": l > hi, "edited": n in edited,
            "window_other": lo <= l <= hi and n not in edited}[part]
sets = load_sets(SETS, mmlu_n=1000)
from concurrent.futures import ThreadPoolExecutor
def set_tensors(get, names, prec_of, chunk=8):
    names = list(names)
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(names), chunk):
            part = names[i:i + chunk]
            outs = list(ex.map(lambda n: roundtrip(get(n), prec_of(n)).to(torch.float16), part))
            with torch.no_grad():
                for n, t in zip(part, outs): params[n].copy_(t)
results = []
for part in PARTS:
    prec_of = lambda n: PREC if in_part(n, part) else "fp16"
    set_tensors(lambda n: master[n], master.keys(), prec_of)
    for name, d in models.items():
        if d: set_tensors(lambda n: d[n], d.keys(), prec_of)
        row = {"model": name, "part": part, "prec": PREC, "window": [lo, hi], "n_rounded": sum(in_part(n, part) for n in master)}
        for k, v in sets.items():
            acc, preds, probs = mcq_eval(model, tok, v)
            row[k] = acc
            row[k + "_pcorrect"] = round(sum(p[a] for p, (_, a) in zip(probs, v)) / len(v), 4)
        results.append(row); log(row)
        if d: set_tensors(lambda n: master[n], d.keys(), prec_of)
        json.dump(results, open("/kaggle/working/locate.json", "w"), indent=1)
print(json.dumps(results, indent=1))

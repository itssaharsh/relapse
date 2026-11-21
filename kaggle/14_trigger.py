# ==== main: does the edit still fire in the file? Score each saved run at fp16 and after 4-bit round trips, and record how strongly
# the residual stream is pushed at the top layer of every edited window (RMU drives it to about 3x its usual size on hazardous text) ====
sh("pip install -q 'bitsandbytes>=0.45'", check=False)
setup_llamacpp(build=False)
from transformers import AutoTokenizer, AutoModelForCausalLM
PRECS = os.environ.get("PRECS", "fp16,Q4_0,nf4").split(",")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "5,7,9,12,15").split(",")]
KEYS = [k for k in os.environ.get("KEYS", "").split(",") if k]      # run names to keep; empty keeps every run in the file
MP = model_path(os.environ.get("VARIANT", "global"))
tok = AutoTokenizer.from_pretrained(MP)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 0}).eval()
params = dict(model.named_parameters())
master = {n: p.detach().cpu().clone() for n, p in params.items() if is_quant_target(n, p)}
models = {"base": {}}
for f in sorted(glob.glob("/kaggle/input/run/*.pt")):
    for k, d in torch.load(f).items():
        if not KEYS or k in KEYS: models[k] = {n: t.float() for n, t in d.items()}
log("runs", list(models))
sets = load_sets(["wmdp-bio"])
items_in = sets["wmdp-bio"]
@torch.no_grad()
def score(bs=8, maxlen=1024):
    """Accuracy, right-or-wrong per question, and per question the size of the residual stream after each layer in LAYERS
    (at the last token, and averaged over the prompt's tokens)."""
    tok.padding_side = "left"; ids = letter_ids(tok)
    order = sorted(range(len(items_in)), key=lambda i: len(items_in[i][0]))
    right = [0] * len(items_in); last = {l: [0.0] * len(items_in) for l in LAYERS}; mean = {l: [0.0] * len(items_in) for l in LAYERS}
    state = {}
    def hook(l):
        def fn(_m, _i, o):
            h = (o[0] if isinstance(o, tuple) else o).float(); n = h.norm(dim=-1); m = state["mask"]
            state[l] = (n[:, -1].cpu(), ((n * m).sum(1) / m.sum(1)).cpu())
        return fn
    hooks = [model.model.layers[l].register_forward_hook(hook(l)) for l in LAYERS]
    try:
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            enc = tok([items_in[i][0] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to("cuda:0")
            state["mask"] = enc["attention_mask"].float()
            p = torch.softmax(model(**enc, logits_to_keep=1).logits[:, -1, :].float()[:, ids], -1).cpu()
            for j, i in enumerate(idx):
                right[i] = int(int(p[j].argmax()) == items_in[i][1])
                for l in LAYERS: last[l][i] = round(float(state[l][0][j]), 2); mean[l][i] = round(float(state[l][1][j]), 2)
    finally:
        for h in hooks: h.remove()
    return sum(right) / len(right), right, last, mean
from concurrent.futures import ThreadPoolExecutor
def set_tensors(get, names, prec, chunk=8):
    names = list(names)
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(names), chunk):
            part = names[i:i + chunk]
            outs = list(ex.map(lambda n: roundtrip(get(n), prec).to(torch.float16), part))
            with torch.no_grad():
                for n, t in zip(part, outs): params[n].copy_(t)
results, items = [], {}
for prec in PRECS:
    set_tensors(lambda n: master[n], master.keys(), prec)
    for name, d in models.items():
        if d: set_tensors(lambda n: d[n], d.keys(), prec)
        acc, right, last, mean = score()
        results.append({"model": name, "prec": prec, "wmdp-bio": acc, "last_norm": {l: round(sum(v) / len(v), 2) for l, v in last.items()}, "mean_norm": {l: round(sum(v) / len(v), 2) for l, v in mean.items()}})
        items[f"{name}|{prec}"] = {"right": right, "last": last, "mean": mean}
        log(results[-1])
        if d: set_tensors(lambda n: master[n], d.keys(), prec)
        json.dump(results, open("/kaggle/working/trigger.json", "w"), indent=1); json.dump(items, open("/kaggle/working/trigger_items.json", "w"))

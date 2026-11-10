# ==== main: second seed. Retrain RMU at the default layers and at the CircuitKIT layers with seed 1, then the same grid ====
sh("pip install -q safetune==0.1.13 'bitsandbytes>=0.45' auditkit==1.1.6", check=False)
setup_llamacpp(build=True)
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from safetune.interventions.unlearn.rmu import rmu_unlearn, RMUConfig
SEED = int(os.environ.get("SEED", "1"))
CFGS = [{"name": "rmu_default_layers_seed1", "layer": 7, "lr": 4e-5, "nb": 250},
        {"name": "rmu_circuit_layers_seed1", "layer": 9, "lr": 1.5e-4, "nb": 300}]
MP = model_path("global")
tok = AutoTokenizer.from_pretrained(MP)
if tok.pad_token is None: tok.pad_token = tok.eos_token
def batches(texts, n, bs=4, maxlen=512):
    tok.padding_side = "right"; out = []
    for i in range(n):
        enc = tok([t[:2000] for t in texts[i * bs:(i + 1) * bs]], return_tensors="pt", padding=True, truncation=True, max_length=maxlen)
        out.append({k: v.to("cuda:0") for k, v in enc.items()})
    return out
def load_trunc(L):
    cfg = AutoConfig.from_pretrained(MP); cfg.num_hidden_layers = L + 1
    if getattr(cfg, "layer_types", None): cfg.layer_types = cfg.layer_types[:L + 1]
    return AutoModelForCausalLM.from_pretrained(MP, config=cfg, dtype=torch.float32, device_map={"": 0}).model.eval()
FB_all = get_texts("bio", 300 * 4, seed=SEED); RB_all = get_texts("wiki", 300 * 4, seed=SEED)
trained = {}
for c in CFGS:
    L = c["layer"]; UPD = [L - 2, L - 1, L]
    FB, RB = batches(FB_all, c["nb"]), batches(RB_all, c["nb"])
    frozen = load_trunc(L)
    for p in frozen.parameters(): p.requires_grad_(False)
    norms = []
    h = frozen.layers[L].register_forward_hook(lambda m, i, o: norms.append((o[0] if isinstance(o, tuple) else o).norm(dim=-1).mean().item()))
    with torch.no_grad():
        for b in FB[:8]: frozen(**b)
    h.remove()
    torch.manual_seed(SEED); random.seed(SEED)
    m = load_trunc(L)
    rmu_unlearn(m, retain_batches=RB, forget_batches=FB, frozen_model=frozen,
                config=RMUConfig(layer_id=L, update_layer_ids=UPD, steering_coeff=3 * sum(norms) / len(norms), alpha=100, lr=c["lr"], max_num_batches=c["nb"]))
    trained[c["name"]] = {"model." + n: p.detach().float().cpu().clone() for n, p in m.named_parameters() if any(f"layers.{l}.mlp.down_proj" in n for l in UPD)}
    torch.save(trained, "/kaggle/working/changed_seed1.pt"); log("trained", c["name"])
    del m, frozen, FB, RB; gc.collect(); torch.cuda.empty_cache()
TAG = "seed1"
SIM = os.environ.get("SIM", "fp16,Q8_0,Q4_0,nf4").split(",")
REAL = os.environ.get("REAL", "Q4_K_M").split(",")
AUDIT = os.environ.get("AUDIT", "fp16,Q4_K_M").split(",")
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 0}).eval()
params = dict(model.named_parameters())
master = {n: p.detach().cpu().clone() for n, p in params.items() if is_quant_target(n, p)}
other = {n: p.detach().cpu().clone() for n, p in params.items() if not is_quant_target(n, p)}
models = {"base": {}}
models.update(trained)
sets = load_sets(["wmdp-bio", "mmlu"], mmlu_n=1000)
from concurrent.futures import ThreadPoolExecutor
def set_tensors(get, names, prec, chunk=8):
    names = list(names)
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(names), chunk):
            part = names[i:i + chunk]
            outs = list(ex.map(lambda n: roundtrip(get(n), prec).to(torch.float16), part))
            with torch.no_grad():
                for n, t in zip(part, outs): params[n].copy_(t)
@torch.no_grad()
def restore_other():
    for n, t in other.items(): params[n].copy_(t)
results, items_out = [], {}
def evaluate(name, prec):
    row = {"model": name, "prec": prec}
    for k, v in sets.items():
        acc, preds, probs = mcq_eval(model, tok, v)
        row[k] = round(acc, 4)
        row[k + "_pcorrect"] = round(sum(p[a] for p, (_, a) in zip(probs, v)) / len(v), 4)
        if k == "wmdp-bio": items_out[f"{name}|{prec}|{k}"] = probs
        elif k == "mmlu": items_out[f"{name}|{prec}|{k}"] = [round(p[a], 4) for p, (_, a) in zip(probs, v)]
    results.append(row); log(row)
    json.dump(results, open(f"/kaggle/working/grid_{TAG}.json", "w"), indent=1)
    json.dump(items_out, open(f"/kaggle/working/items_{TAG}.json", "w"))
# --- simulated / exact round trips ---
for prec in SIM:
    restore_other()
    set_tensors(lambda n: master[n], master.keys(), prec)
    for name, d in models.items():
        if d: set_tensors(lambda n: d[n], d.keys(), prec)
        evaluate(name, prec)
        if d: set_tensors(lambda n: master[n], d.keys(), prec)
# --- real llama.cpp k-quants (all tensors incl. embeddings, as llama.cpp stores them) ---
hf_to_gguf(MP, "/tmp/base-f16.gguf", "f16")
import gguf
from gguf import GGUFReader
nm = {v: k for k, v in gguf_name_map(model).items()}   # hf name -> gguf name
for name, d in models.items():
    src = "/tmp/base-f16.gguf"
    if d:
        src = f"/tmp/{name}-f16.gguf"; shutil.copyfile("/tmp/base-f16.gguf", src)
        rd = GGUFReader(src, "r+")
        tens = {t.name: t for t in rd.tensors}
        for n, t in d.items():
            g = tens[nm[n]]
            assert tuple(g.data.shape) == tuple(t.shape), (g.data.shape, t.shape)
            g.data[...] = t.numpy().astype("float16")
        rd.data.flush(); del rd
    for qt in REAL:
        dst = f"/tmp/{name}-{qt}.gguf"; quantize_gguf(src, dst, qt)
        load_gguf_into(model, dst)
        evaluate(name, qt)
        if name != "base" and qt == "Q4_K_M": shutil.copyfile(dst, f"/kaggle/working/tiny-aya-global-{name}-Q4_K_M.gguf")
        if name == "base" or qt != "Q4_K_M": os.remove(dst)
    restore_other()
# --- AuditKIT: paired comparisons with bootstrap CIs (forget holds? price of unlearning?) ---
try:
    exec(open(glob.glob("/kaggle/input/**/auditkit_mcq.py", recursive=True)[0]).read(), globals())
    runs = {}
    for name, d in models.items():
        for prec in AUDIT:
            if prec in REAL:
                path = f"/tmp/{name}-{prec}.gguf"
                if not os.path.exists(path): quantize_gguf("/tmp/base-f16.gguf" if name == "base" else f"/tmp/{name}-f16.gguf", path, prec)
                load_gguf_into(model, path)
            else:
                restore_other(); set_tensors(lambda n: master[n], master.keys(), prec)
                if d: set_tensors(lambda n: d[n], d.keys(), prec)
            for k in ("wmdp-bio", "mmlu"):
                r = audit_mcq(model, tok, sets[k], f"{name}-{prec}-{k}")
                r.save(f"/kaggle/working/audit_{TAG}_{name}_{prec}_{k}.json"); runs[(name, prec, k)] = r
    audits = []
    lo, hi = AUDIT[0], AUDIT[-1]
    for name in models:
        c = audit_compare(runs[(name, lo, "wmdp-bio")], runs[(name, hi, "wmdp-bio")]); c.pop("summary", None)
        c.update({"model": name, "kind": "forget_holds", "from": lo, "to": hi}); audits.append(c); log("audit", c)
        if name != "base":
            for prec in (lo, hi):
                c = audit_compare(runs[("base", prec, "mmlu")], runs[(name, prec, "mmlu")]); c.pop("summary", None)
                c.update({"model": name, "kind": "price", "prec": prec}); audits.append(c); log("audit", c)
    json.dump(audits, open(f"/kaggle/working/audit_{TAG}.json", "w"), indent=1)
except Exception as e:
    import traceback; traceback.print_exc(); log("AUDIT FAILED", repr(e)[:400])
print(json.dumps(results, indent=1))

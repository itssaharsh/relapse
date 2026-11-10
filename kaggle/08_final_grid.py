# ==== main: final evaluation grid (base + unlearned models) x storage formats x eval sets ====
sh("pip install -q 'bitsandbytes>=0.45' auditkit==1.1.6", check=False)
setup_llamacpp(build=True)
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
TAG = os.environ.get("TAG", "final")
RUNS = json.loads(os.environ.get("RUNS", "[]"))   # [{"name": "...", "file": "changed_rmu2.pt", "key": "L9ck_a100_lr1e-4"}]
SIM = os.environ.get("SIM", "fp16,bf16,Q8_0,Q4_0,nf4").split(",")
REAL = os.environ.get("REAL", "Q6_K,Q4_K_M").split(",")
AUDIT = os.environ.get("AUDIT", "fp16,Q4_K_M").split(",")
MP = model_path("global")
tok = AutoTokenizer.from_pretrained(MP)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 0}).eval()
params = dict(model.named_parameters())
master = {n: p.detach().cpu().clone() for n, p in params.items() if is_quant_target(n, p)}
other = {n: p.detach().cpu().clone() for n, p in params.items() if not is_quant_target(n, p)}
models = {"base": {}}
for r in RUNS:
    f = glob.glob(f"/kaggle/input/**/{r['file']}", recursive=True)[0]
    models[r["name"]] = {n: t.float() for n, t in torch.load(f)[r["key"]].items()}
    log("loaded run", r["name"], "from", f, list(models[r["name"]])[:3])
# eval sets
sets = load_sets(["wmdp-bio", "mmlu"], mmlu_n=1000)
pf = glob.glob("/kaggle/input/**/probes.json", recursive=True)
probes = json.load(open(pf[0])) if pf else []
for lang in sorted({p["lang"] for p in probes}):
    sets["probe:" + lang] = [("The following are multiple choice questions (with answers) about biology.\n\n" + p["item"] + "\nAnswer:", p["answer"]) for p in probes if p["lang"] == lang]
log("sets", {k: len(v) for k, v in sets.items()})
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

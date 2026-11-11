# ==== main: RMU re-tune sweep (SafeTune) with loss logs; fp16 screening eval ====
sh("pip install -q safetune==0.1.13")
setup_llamacpp(build=False)
import logging
logging.basicConfig(level=logging.WARNING)
lg = logging.getLogger("safetune.interventions.unlearn.rmu"); lg.setLevel(logging.INFO)
class _Every(logging.Filter):
    def filter(self, rec):
        m = re.search(r"step (\d+)", rec.getMessage()); return bool(m) and int(m.group(1)) % 25 == 0
for h in logging.getLogger().handlers: h.addFilter(_Every())
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from safetune.interventions.unlearn.rmu import rmu_unlearn, RMUConfig
CONFIGS = json.loads(os.environ.get("CONFIGS", json.dumps([
    {"name": "L9_lr1.5e-4", "layer": 9, "alpha": 100, "lr": 1.5e-4, "mult": 3},
    {"name": "L9_a300_lr2e-4", "layer": 9, "alpha": 300, "lr": 2e-4, "mult": 3},
    {"name": "L9_lr1.5e-4_qa", "layer": 9, "alpha": 100, "lr": 1.5e-4, "mult": 3, "pq": 0.5}])))
NB = int(os.environ.get("NB", "300"))
import torch.nn.utils.parametrize as parametrize
PRECS = os.environ.get("PRECS", "fp16,Q4_0,nf4").split(",")
random.seed(0); torch.manual_seed(0)
MP = model_path(os.environ.get("VARIANT", "global"))
tok = AutoTokenizer.from_pretrained(MP)
if tok.pad_token is None: tok.pad_token = tok.eos_token
def batches(texts, n, bs=4, maxlen=512):
    tok.padding_side = "right"; out = []
    for i in range(n):
        enc = tok([t[:2000] for t in texts[i * bs:(i + 1) * bs]], return_tensors="pt", padding=True, truncation=True, max_length=maxlen)
        out.append({k: v.to("cuda:0") for k, v in enc.items()})
    return out
_BATCHES = {}
def batches_for(seed):   # a config may carry its own "seed" (data order and the RMU control vector)
    if seed not in _BATCHES:
        _BATCHES.clear()
        _BATCHES[seed] = (batches(get_texts("bio", NB * 4, seed=seed), NB), batches(get_texts("wiki", NB * 4, seed=seed), NB))
    return _BATCHES[seed]
FROZEN = torch.float16 if os.environ.get("FROZEN_FP16") == "1" else torch.float32   # fp16 frozen reference: room for deeper windows on a T4
def load_trunc(L, dtype=torch.float32):
    cfg = AutoConfig.from_pretrained(MP); cfg.num_hidden_layers = L + 1
    if getattr(cfg, "layer_types", None): cfg.layer_types = cfg.layer_types[:L + 1]
    return AutoModelForCausalLM.from_pretrained(MP, config=cfg, dtype=dtype, device_map={"": 0}).model.eval()
changed = {}
for c in CONFIGS:
    L = c["layer"]; UPD = [L - 2, L - 1, L]; SD = c.get("seed", 0)
    FB, RB = batches_for(SD)
    frozen = load_trunc(L, FROZEN)
    for p in frozen.parameters(): p.requires_grad_(False)
    norms = []
    h = frozen.layers[L].register_forward_hook(lambda m, i, o: norms.append((o[0] if isinstance(o, tuple) else o).norm(dim=-1).mean().item()))
    with torch.no_grad():
        for b in FB[:8]: frozen(**b)
    h.remove(); anorm = sum(norms) / len(norms)
    model = load_trunc(L)
    torch.manual_seed(SD); random.seed(SD)
    if c.get("pq"):
        for l in UPD:
            parametrize.register_parametrization(model.layers[l].mlp.down_proj, "weight", FakeQuantSTE(p=c["pq"]))
    detach = None
    if c.get("pq_file"):   # the whole truncated model, and the frozen reference, seen as the Q4_0 file on a fraction of the steps
        detach = attach_file_view(model, frozen, {f"layers.{l}.mlp.down_proj" for l in UPD}, c["pq_file"], seed=SD,
                                  qdev="cuda:1" if torch.cuda.device_count() > 1 else None)
    rc = RMUConfig(layer_id=L, update_layer_ids=UPD, steering_coeff=c["mult"] * anorm, alpha=c["alpha"], lr=c["lr"], max_num_batches=NB)
    t1 = time.time()
    rmu_unlearn(model, retain_batches=RB, forget_batches=FB, frozen_model=frozen, config=rc)
    if c.get("pq"):
        for l in UPD:
            parametrize.remove_parametrizations(model.layers[l].mlp.down_proj, "weight", leave_parametrized=False)
    if detach: log("file view", detach())
    changed[c["name"]] = {"model." + n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if any(f"layers.{l}.mlp.down_proj" in n for l in UPD)}
    log("trained", c["name"], "anorm", round(anorm, 2), "secs", round(time.time() - t1))
    torch.save(changed, "/kaggle/working/" + os.environ.get("OUT", "changed_rmu4.pt"))
    del model, frozen; gc.collect(); torch.cuda.empty_cache()
del FB, RB; _BATCHES.clear(); gc.collect(); torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 0}).eval()
params = dict(model.named_parameters())
master = {n: p.detach().cpu().clone() for n, p in params.items() if is_quant_target(n, p)}
sets = load_sets(["wmdp-bio", "mmlu"], mmlu_n=1000)
pf = glob.glob("/kaggle/input/**/probes.json", recursive=True)
if pf:
    probes = json.load(open(pf[0]))
    for lang in ["en_para", "hi", "sw"]:
        sets["probe:" + lang] = [("The following are multiple choice questions (with answers) about biology.\n\n" + p["item"] + "\nAnswer:", p["answer"]) for p in probes if p["lang"] == lang]
from concurrent.futures import ThreadPoolExecutor
def set_tensors(get, names, prec, chunk=8):
    names = list(names)
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(names), chunk):
            part = names[i:i + chunk]
            outs = list(ex.map(lambda n: roundtrip(get(n), prec).to(torch.float16), part))
            with torch.no_grad():
                for n, t in zip(part, outs): params[n].copy_(t)
results = []
for prec in PRECS:
    set_tensors(lambda n: master[n], master.keys(), prec)
    for name, d in ([] if os.environ.get("SKIP_BASE") == "1" else [("base", {})]) + list(changed.items()):
        if d: set_tensors(lambda n: d[n], d.keys(), prec)
        row = {"model": name, "prec": prec}
        for k, v in sets.items():
            acc, preds, probs = mcq_eval(model, tok, v)
            row[k] = acc
            row[k + "_pcorrect"] = round(sum(p[a] for p, (_, a) in zip(probs, v)) / len(v), 4)
        results.append(row); log(row)
        if d: set_tensors(lambda n: master[n], d.keys(), prec)
        json.dump(results, open("/kaggle/working/results.json", "w"), indent=1)
torch.save(changed, "/kaggle/working/" + os.environ.get("OUT", "changed_rmu4.pt"))
print(json.dumps(results, indent=1))

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
    {"name": "L7_a100_lr1e-4", "layer": 7, "alpha": 100, "lr": 1e-4, "mult": 3},
    {"name": "L7_a10_lr1e-4", "layer": 7, "alpha": 10, "lr": 1e-4, "mult": 3},
    {"name": "L9ck_a100_lr1e-4", "layer": 9, "alpha": 100, "lr": 1e-4, "mult": 3},
    {"name": "L7_a100_lr3e-4", "layer": 7, "alpha": 100, "lr": 3e-4, "mult": 3}])))
NB = int(os.environ.get("NB", "300"))
random.seed(0); torch.manual_seed(0)
MP = model_path("global")
tok = AutoTokenizer.from_pretrained(MP)
if tok.pad_token is None: tok.pad_token = tok.eos_token
forget_txt = get_texts("bio", NB * 4); retain_txt = get_texts("wiki", NB * 4)
def batches(texts, n, bs=4, maxlen=512):
    tok.padding_side = "right"; out = []
    for i in range(n):
        enc = tok([t[:2000] for t in texts[i * bs:(i + 1) * bs]], return_tensors="pt", padding=True, truncation=True, max_length=maxlen)
        out.append({k: v.to("cuda:0") for k, v in enc.items()})
    return out
FB = batches(forget_txt, NB); RB = batches(retain_txt, NB)
def load_trunc(L):
    cfg = AutoConfig.from_pretrained(MP); cfg.num_hidden_layers = L + 1
    if getattr(cfg, "layer_types", None): cfg.layer_types = cfg.layer_types[:L + 1]
    return AutoModelForCausalLM.from_pretrained(MP, config=cfg, dtype=torch.float32, device_map={"": 0}).model.eval()
changed = {}
for c in CONFIGS:
    L = c["layer"]; UPD = [L - 2, L - 1, L]
    frozen = load_trunc(L)
    for p in frozen.parameters(): p.requires_grad_(False)
    norms = []
    h = frozen.layers[L].register_forward_hook(lambda m, i, o: norms.append((o[0] if isinstance(o, tuple) else o).norm(dim=-1).mean().item()))
    with torch.no_grad():
        for b in FB[:8]: frozen(**b)
    h.remove(); anorm = sum(norms) / len(norms)
    model = load_trunc(L)
    rc = RMUConfig(layer_id=L, update_layer_ids=UPD, steering_coeff=c["mult"] * anorm, alpha=c["alpha"], lr=c["lr"], max_num_batches=NB)
    t1 = time.time()
    rmu_unlearn(model, retain_batches=RB, forget_batches=FB, frozen_model=frozen, config=rc)
    changed[c["name"]] = {"model." + n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if any(f"layers.{l}.mlp.down_proj" in n for l in UPD)}
    log("trained", c["name"], "anorm", round(anorm, 2), "secs", round(time.time() - t1))
    del model, frozen; gc.collect(); torch.cuda.empty_cache()
del FB, RB; gc.collect(); torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 0}).eval()
params = dict(model.named_parameters())
master = {n: params[n].detach().cpu().clone() for d in changed.values() for n in d}
sets = load_sets(["wmdp-bio", "mmlu"], mmlu_n=1000)
results = []
for name, d in [("base", {})] + list(changed.items()):
    with torch.no_grad():
        for n, t in d.items(): params[n].copy_(t.half())
    row = {"model": name, "prec": "fp16"}
    for k, v in sets.items(): row[k] = mcq_eval(model, tok, v)[0]
    if d:
        row["mean_abs_delta"] = float(torch.stack([(t - master[n].float()).abs().mean() for n, t in d.items()]).mean())
        # quick Q4_0 check of the changed tensors only (base tensors stay fp16)
        with torch.no_grad():
            for n, t in d.items(): params[n].copy_(roundtrip(t, "Q4_0").half())
        row["wmdp-bio_Q4_0_changed_only"] = mcq_eval(model, tok, sets["wmdp-bio"])[0]
    results.append(row); log(row)
    with torch.no_grad():
        for n in d: params[n].copy_(master[n])
    json.dump(results, open("/kaggle/working/results.json", "w"), indent=1)
torch.save(changed, "/kaggle/working/changed_rmu2.pt")
print(json.dumps(results, indent=1))

# ==== main: pipeline shift. Train RMU through one tokenizer setting (or both), score through each ====
# "official" is the tokenizer Tiny Aya was trained with. "outdated" is the setting in Kaggle's copy under transformers v5.
setup_llamacpp(build=False)
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from safetune.interventions.unlearn.rmu import rmu_unlearn, RMUConfig
CONFIGS = json.loads(os.environ["CONFIGS"])
NB = int(os.environ.get("NB", "300"))
MP = model_path(os.environ["VARIANT"])
def outdated_tokenizer(mp):
    d = "/tmp/outdated_tok"; os.makedirs(d, exist_ok=True)
    for f in ("tokenizer.json", "special_tokens_map.json"): shutil.copy(f"{mp}/{f}", d)
    c = json.load(open(f"{mp}/tokenizer_config.json")); c["tokenizer_class"] = "CohereTokenizerFast"
    json.dump(c, open(f"{d}/tokenizer_config.json", "w"))
    return AutoTokenizer.from_pretrained(d)
TOKS = {"official": AutoTokenizer.from_pretrained(MP), "outdated": outdated_tokenizer(MP)}
log({k: type(v).__name__ for k, v in TOKS.items()})
for t in TOKS.values():
    if t.pad_token is None: t.pad_token = t.eos_token
def batches(texts, n, which, bs=4, maxlen=512):
    out = []
    for i in range(n):
        t = TOKS[which] if which != "both" else TOKS["official" if i % 2 == 0 else "outdated"]    # the pair (forget, retain) of a step shares a tokenizer
        t.padding_side = "right"
        enc = t([x[:2000] for x in texts[i * bs:(i + 1) * bs]], return_tensors="pt", padding=True, truncation=True, max_length=maxlen)
        out.append({k: v.to("cuda:0") for k, v in enc.items()})
    return out
def load_trunc(L, dtype=torch.float32):
    cfg = AutoConfig.from_pretrained(MP); cfg.num_hidden_layers = L + 1
    if getattr(cfg, "layer_types", None): cfg.layer_types = cfg.layer_types[:L + 1]
    return AutoModelForCausalLM.from_pretrained(MP, config=cfg, dtype=dtype, device_map={"": 0}).model.eval()
changed, TEXTS, CACHE = {}, {}, {}
for c in CONFIGS:
    L = c["layer"]; UPD = [L - 2, L - 1, L]; SD = c["seed"]; W = c["train_tok"]
    if SD not in TEXTS: TEXTS[SD] = (get_texts("bio", NB * 4, seed=SD), get_texts("wiki", NB * 4, seed=SD))
    if W not in CACHE: CACHE.clear(); CACHE[W] = (batches(TEXTS[SD][0], NB, W), batches(TEXTS[SD][1], NB, W))
    FB, RB = CACHE[W]
    frozen = load_trunc(L, torch.float16)
    for p in frozen.parameters(): p.requires_grad_(False)
    norms = []
    h = frozen.layers[L].register_forward_hook(lambda m, i, o: norms.append((o[0] if isinstance(o, tuple) else o).norm(dim=-1).mean().item()))
    with torch.no_grad():
        for b in FB[:8]: frozen(**b)
    h.remove(); anorm = sum(norms) / len(norms)
    model = load_trunc(L)
    torch.manual_seed(SD); random.seed(SD)
    rc = RMUConfig(layer_id=L, update_layer_ids=UPD, steering_coeff=c["mult"] * anorm, alpha=c["alpha"], lr=c["lr"], max_num_batches=NB)
    t1 = time.time()
    rmu_unlearn(model, retain_batches=RB, forget_batches=FB, frozen_model=frozen, config=rc)
    changed[c["name"]] = {"model." + n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if any(f"layers.{l}.mlp.down_proj" in n for l in UPD)}
    log("trained", c["name"], "anorm", round(anorm, 2), "secs", round(time.time() - t1))
    del model, frozen; gc.collect(); torch.cuda.empty_cache()
CACHE.clear(); gc.collect(); torch.cuda.empty_cache()
torch.save(changed, "/kaggle/working/" + os.environ["OUT"])
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 0}).eval()
params = dict(model.named_parameters())
master = {n: p.detach().cpu().clone() for n, p in params.items() if is_quant_target(n, p)}
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
results, items = [], {}
for prec, score_toks in (("fp16", ("official", "outdated")), ("Q4_0", ("official",)), ("nf4", ("official",))):
    set_tensors(lambda n: master[n], master.keys(), prec)
    for name, d in [("base", {})] + list(changed.items()):
        if d: set_tensors(lambda n: d[n], d.keys(), prec)
        for tn in score_toks:
            row = {"model": name, "prec": prec, "score_tok": tn}
            for k, v in sets.items():
                acc, preds, probs = mcq_eval(model, TOKS[tn], v)
                row[k] = acc
                items[f"{name}|{prec}|{tn}|{k}"] = [int(x == y[1]) for x, y in zip(preds, v)]
            results.append(row); log(row)
            json.dump(results, open("/kaggle/working/results.json", "w"), indent=1); json.dump(items, open("/kaggle/working/items.json", "w"))
        if d: set_tensors(lambda n: master[n], d.keys(), prec)

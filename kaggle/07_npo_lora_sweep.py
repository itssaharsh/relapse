# ==== main: NPO (SafeTune, LoRA) unlearning + precision grid ====
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sh("pip install -q safetune==0.1.13 'bitsandbytes>=0.45' && pip uninstall -y -q torchao")
setup_llamacpp(build=False)
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetune.interventions.unlearn.npo import npo_unlearn, NPOConfig
from safetune.runner.utils.model_utils import lora_wrap
LRS = [float(x) for x in os.environ.get("LRS", "2e-4,5e-4,1e-3").split(",")]
STEPS = int(os.environ.get("STEPS", "250")); BS = 1; MAXLEN = 256
PRECS = os.environ.get("PRECS", "bf16,Q8_0,Q4_0,nf4").split(",")
MP = model_path("global")
tok = AutoTokenizer.from_pretrained(MP)
if tok.pad_token is None: tok.pad_token = tok.eos_token
def mk_batches(texts, dev="cuda:0"):
    tok.padding_side = "right"; out = []
    for i in range(0, len(texts), BS):
        enc = tok([t[:2000] for t in texts[i:i + BS]], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN)
        lab = enc["input_ids"].clone(); lab[enc["attention_mask"] == 0] = -100
        enc["labels"] = lab
        out.append({k: v.to(dev) for k, v in enc.items()})
    return out
FB = mk_batches(get_texts("bio", STEPS * BS)); RB = mk_batches(get_texts("wiki", STEPS * BS))
ref = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 1}).eval()
for p in ref.parameters(): p.requires_grad_(False)
def fwd(m, batch):
    dev = next(m.parameters()).device
    b = {k: v.to(dev) for k, v in batch.items() if k != "labels"}
    return m(**b).logits.float().to("cuda:0")
sets = load_sets(["wmdp-bio", "mmlu"], mmlu_n=1000)
base_master = None
deltas = {}; fp16_rows = []
for lr in LRS:
    name = f"npo_lr{lr}"
    pol = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 0})
    if base_master is None:
        base_master = {n: p.detach().cpu().clone() for n, p in pol.named_parameters() if is_quant_target(n, p)}
    pol.config.use_cache = False
    pol.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pol.enable_input_require_grads()
    pol = lora_wrap(pol, r=16, lora_alpha=32, lora_dropout=0.0)
    t1 = time.time()
    npo_unlearn(pol, FB, RB, reference=ref, forward_fn=fwd, config=NPOConfig(variant="npo_grad_diff", beta=0.1, num_epochs=1, lr=lr, max_steps=STEPS))
    # exact fp32 deltas from the LoRA factors (no fp16 merge rounding)
    d = {}
    for mn, mod in pol.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            A = mod.lora_A["default"].weight.detach().float(); B = mod.lora_B["default"].weight.detach().float()
            sc = mod.scaling["default"]
            hf = mn.replace("base_model.model.", "") + ".weight"
            d[hf] = (B.cpu(), A.cpu(), float(sc))
    deltas[name] = d
    log("trained", name, "secs", round(time.time() - t1), "n_delta", len(d), "mean|d|", float(sum((B @ A * sc).abs().mean() for B, A, sc in d.values()) / len(d)))
    pol = pol.unload()
    # fp16 eval of base+delta (weights recomputed in fp32 then cast)
    params = dict(pol.named_parameters())
    with torch.no_grad():
        for n, (B, A, sc) in d.items(): params[n].copy_((base_master[n].float() + B @ A * sc).to(torch.float16))
    pol.eval(); row = {"prec": "fp16", "model": name}
    for k, v in sets.items(): row[k] = mcq_eval(pol, tok, v)[0]
    fp16_rows.append(row); log(row)
    del pol, params; gc.collect(); torch.cuda.empty_cache()
del ref; gc.collect(); torch.cuda.empty_cache()
# pick configs that keep MMLU within 3 points of base (base mmlu measured in RMU run ~ use best guess: max over rows)
results = list(fp16_rows)
json.dump(results, open("/kaggle/working/results.json", "w"), indent=1)
torch.save(deltas, "/kaggle/working/npo_deltas.pt")
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 0}).eval()
params = dict(model.named_parameters())
from concurrent.futures import ThreadPoolExecutor
def set_tensors(get, names, prec, chunk=8):
    names = list(names)
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(names), chunk):
            part = names[i:i + chunk]
            outs = list(ex.map(lambda n: roundtrip(get(n), prec).to(torch.float16), part))
            with torch.no_grad():
                for n, t in zip(part, outs): params[n].copy_(t)
for prec in PRECS:
    try:
        set_tensors(lambda n: base_master[n], base_master.keys(), prec)
    except Exception as e:
        log("SKIP", prec, repr(e)[:200]); continue
    row = {"prec": prec, "model": "base"}
    for k, v in sets.items(): row[k] = mcq_eval(model, tok, v)[0]
    results.append(row); log(row)
    for name, d in deltas.items():
        set_tensors(lambda n: base_master[n].float() + d[n][0] @ d[n][1] * d[n][2], d.keys(), prec)
        row = {"prec": prec, "model": name}
        for k, v in sets.items(): row[k] = mcq_eval(model, tok, v)[0]
        results.append(row); log(row)
        set_tensors(lambda n: base_master[n], d.keys(), prec)
    json.dump(results, open("/kaggle/working/results.json", "w"), indent=1)
print(json.dumps(results, indent=1))

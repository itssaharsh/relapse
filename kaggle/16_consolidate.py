# ==== main: file consolidation. Take a finished unlearned model and retrain only its three edited matrices so that the model,
# seen as its 4-bit file, behaves like the full-precision model that was audited. ====
# Teacher: the unlearned model at full precision. Student: the same model with every decoder linear weight replaced by its Q4_0
# round trip; the edited matrices are rounded on the fly with a straight-through gradient. Loss: distance between the student's and
# the teacher's final hidden states, on forget text and on retain text.
setup_llamacpp(build=False)
import torch.nn.utils.parametrize as P
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
KEYS = [k for k in os.environ["KEYS"].split(",") if k]
STEPS = int(os.environ.get("STEPS", "200")); LRS = [float(x) for x in os.environ.get("LRS", "3e-5").split(",")]
BS = int(os.environ.get("BS", "4")); MAXLEN = int(os.environ.get("MAXLEN", "256")); DATA_SEED = int(os.environ.get("DATA_SEED", "7"))
MP = model_path(os.environ["VARIANT"])
src = torch.load(hf_hub_download("itssaharsh/relapse-stage2", os.environ["SRC_PT"], repo_type="dataset"))
tok = AutoTokenizer.from_pretrained(MP)
if tok.pad_token is None: tok.pad_token = tok.eos_token
def batches(texts, n):
    tok.padding_side = "right"; out = []
    for i in range(n):
        enc = tok([t[:2000] for t in texts[i * BS:(i + 1) * BS]], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN)
        out.append({k: v.to("cuda:0") for k, v in enc.items()})
    return out
FB = batches(get_texts("bio", STEPS * BS, seed=DATA_SEED), STEPS); RB = batches(get_texts("wiki", STEPS * BS, seed=DATA_SEED), STEPS)
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float32, device_map={"": 0}).eval()
for p in model.parameters(): p.requires_grad_(False)
params = dict(model.named_parameters())
master = {n: p.detach().half().cpu().clone() for n, p in params.items() if is_quant_target(n, p)}      # the original weights, as stored
mods = dict(model.named_modules())
class FileEdit(torch.nn.Module):
    """A trained matrix as the file stores it: Q4_0 round trip on the fly, straight-through gradient."""
    def __init__(self, w0):
        super().__init__(); self.w = torch.nn.Parameter(w0.float().clone())
    def forward(self, W):
        return self.w + (fq_q4_0(self.w.detach()) - self.w.detach())
def hidden(batch):
    return model.model(**batch).last_hidden_state
def dist(h, target, mask):
    m = mask.unsqueeze(-1).float()
    return (((h - target) ** 2) * m).sum() / ((target ** 2) * m).sum()
changed = {}
for key in KEYS:
    edit = {n: t.float() for n, t in src[key].items()}
    for lr in LRS:
        name = re.sub(r"_s(\d+)$", rf"_cons_lr{lr:g}_s\1", key)
        with torch.no_grad():                                           # teacher: full precision, edited
            for n, t in master.items(): params[n].copy_(t.float())
            for n, t in edit.items(): params[n].copy_(t.half().float())
            TF = [hidden(b).half().cpu() for b in FB]; TR = [hidden(b).half().cpu() for b in RB]
            for n, t in master.items(): params[n].copy_(fq_q4_0(t.float().cuda()))      # student: every decoder linear weight as the file stores it
        fe = {}
        for n, t in edit.items():
            mod = mods[n[:-len(".weight")]]; fe[n] = FileEdit(t.half().float().cuda())
            P.register_parametrization(mod, "weight", fe[n], unsafe=True)
        opt = torch.optim.AdamW([f.w for f in fe.values()], lr=lr, weight_decay=0.0)
        t1 = time.time()
        for step in range(STEPS):
            loss_f = dist(hidden(FB[step]), TF[step].cuda().float(), FB[step]["attention_mask"])
            loss_r = dist(hidden(RB[step]), TR[step].cuda().float(), RB[step]["attention_mask"])
            loss = loss_f + loss_r
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 25 == 0 or step == STEPS - 1: log(name, "step", step, "forget", round(float(loss_f), 5), "retain", round(float(loss_r), 5))
        changed[name] = {n: f.w.detach().float().cpu().clone() for n, f in fe.items()}
        moved = {n.split(".layers.")[1].split(".")[0]: round(float((changed[name][n] - edit[n]).norm() / edit[n].norm()), 5) for n in edit}
        log("consolidated", name, "secs", round(time.time() - t1), "relative change per layer", moved)
        for n in edit: P.remove_parametrizations(mods[n[:-len(".weight")]], "weight", leave_parametrized=False)
        del TF, TR, fe, opt; gc.collect(); torch.cuda.empty_cache()
    changed[key] = edit                                                  # the plain run, scored in the same job for paired comparisons
torch.save({k: v for k, v in changed.items() if "_cons_" in k}, "/kaggle/working/" + os.environ.get("OUT", "changed_cons.pt"))
del FB, RB; gc.collect(); torch.cuda.empty_cache()
model = model.half()
params = dict(model.named_parameters())
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
for prec in os.environ.get("PRECS", "fp16,Q4_0,nf4").split(","):
    set_tensors(lambda n: master[n], master.keys(), prec)
    for name, d in [("base", {})] + list(changed.items()):
        if d: set_tensors(lambda n: d[n], d.keys(), prec)
        row = {"model": name, "prec": prec}
        for k, v in sets.items():
            acc, preds, probs = mcq_eval(model, tok, v)
            row[k] = acc; items[f"{name}|{prec}|{k}"] = [int(x == y[1]) for x, y in zip(preds, v)]
        results.append(row); log(row)
        if d: set_tensors(lambda n: master[n], d.keys(), prec)
        json.dump(results, open("/kaggle/working/results.json", "w"), indent=1); json.dump(items, open("/kaggle/working/items.json", "w"))

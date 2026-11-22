# ==== main: NF4 as people really load it. Save each model to disk, load it through bitsandbytes in 4-bit NF4, and score it,
# next to the same weights at fp16. This is the deployed form of NF4; the round trip used elsewhere imitates it. ====
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
KEYS = [k for k in os.environ.get("KEYS", "").split(",") if k]
MP = model_path(os.environ.get("VARIANT", "global"))
tok = AutoTokenizer.from_pretrained(MP)
if tok.pad_token is None: tok.pad_token = tok.eos_token
edits = {"base": {}}
for f in sorted(glob.glob("/kaggle/input/run/*.pt")):
    for k, d in torch.load(f).items():
        if not KEYS or k in KEYS: edits[k] = {n: t.float() for n, t in d.items()}
sets = load_sets(["wmdp-bio", "mmlu"], mmlu_n=1000)
results, items = [], {}
def score(model, name, prec):
    row = {"model": name, "prec": prec}
    for k, v in sets.items():
        acc, preds, _ = mcq_eval(model, tok, v)
        row[k] = acc; items[f"{name}|{prec}|{k}"] = [int(x == y[1]) for x, y in zip(preds, v)]
    results.append(row); log(row)
    json.dump(results, open("/kaggle/working/bnb.json", "w"), indent=1); json.dump(items, open("/kaggle/working/bnb_items.json", "w"))
for name, d in edits.items():
    model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 0}).eval()
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in d.items(): params[n].copy_(t.half())
    score(model, name, "fp16")
    out = f"/tmp/saved_{name}"; model.save_pretrained(out); tok.save_pretrained(out)
    del model, params; gc.collect(); torch.cuda.empty_cache()
    q = AutoModelForCausalLM.from_pretrained(out, device_map={"": 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()
    log("4-bit modules", sum(type(m).__name__ == "Linear4bit" for m in q.modules()))
    score(q, name, "nf4_bitsandbytes")
    del q; gc.collect(); torch.cuda.empty_cache(); shutil.rmtree(out)

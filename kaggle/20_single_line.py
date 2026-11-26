# ==== main: pre-registration 6. Rescore the saved runs of pre-registration 5 with single-line prompts, through both tokenizer settings ====
# Scoring only: the edited matrices come from the jobs of pre-registration 5; nothing is trained here.
import shutil
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
MP = model_path(V)
def outdated_tokenizer(mp):      # as kaggle/13_pipeline_shift.py: the official files with the class name Kaggle's copy carries
    d = "/tmp/outdated_tok"; os.makedirs(d, exist_ok=True)
    for f in ("tokenizer.json", "special_tokens_map.json"): shutil.copy(f"{mp}/{f}", d)
    c = json.load(open(f"{mp}/tokenizer_config.json")); c["tokenizer_class"] = "CohereTokenizerFast"
    json.dump(c, open(f"{d}/tokenizer_config.json", "w"))
    return AutoTokenizer.from_pretrained(d)
TOKS = {"official": AutoTokenizer.from_pretrained(MP), "outdated": outdated_tokenizer(MP)}
log({k: type(v).__name__ for k, v in TOKS.items()}, "model revision", MODEL_REV)
assert type(TOKS["outdated"]).__name__ == "CohereTokenizer" and type(TOKS["official"]).__name__ != "CohereTokenizer"
one = lambda s: " ".join(str(s).split())      # every run of white space, line breaks included, becomes one space
def fmt_single(subject, q, choices):
    return f"The following are multiple choice questions (with answers) about {subject}. {one(q)} " + " | ".join(f"{l}. {one(c)}" for l, c in zip(LET, choices)) + " Answer:"
wmdp = load_dataset("cais/wmdp", "wmdp-bio", split="test", revision=WMDP_REV)
mmlu = load_dataset("cais/mmlu", "all", split="test", revision=MMLU_REV).shuffle(seed=0).select(range(1000))
SETS = {"single": {"wmdp-bio": [(fmt_single("biology", r["question"], r["choices"]), r["answer"]) for r in wmdp],
                   "mmlu": [(fmt_single(r["subject"].replace("_", " "), r["question"], r["choices"]), r["answer"]) for r in mmlu]},
        "mcq": {"wmdp-bio": [(fmt_mcq("biology", r["question"], r["choices"]), r["answer"]) for r in wmdp]}}      # the registered layout, scored again as a check
assert not any("\n" in p for p, _ in SETS["single"]["wmdp-bio"] + SETS["single"]["mmlu"])
changed = {}
for s, path in PT.items():
    changed.update(torch.load(path, map_location="cpu"))
log("runs", len(changed))
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 0}).eval()
params = dict(model.named_parameters())
master = {n: params[n].detach().cpu().clone() for d in changed.values() for n in d}
def put(tensors):
    with torch.no_grad():
        for n, t in tensors.items(): params[n].copy_(t.to(torch.float16))
results, items = [], {}
for name, d in [("base", {})] + sorted(changed.items()):
    if d: put(d)
    for layout in ("single", "mcq"):
        if layout == "mcq" and name != "base" and "_W5-7_" not in name:      # the check of the registered layout: the original and the default-layer runs
            continue
        for tn in ("official", "outdated"):
            row = {"model": name, "prec": "fp16", "score_tok": tn, "layout": layout}
            for k, v in SETS[layout].items():
                acc, preds, probs = mcq_eval(model, TOKS[tn], v)
                row[k] = acc
                items[f"{name}|fp16|{tn}|{layout}|{k}"] = [int(x == y[1]) for x, y in zip(preds, v)]
            results.append(row); log(row)
            json.dump(results, open("/kaggle/working/results.json", "w"), indent=1); json.dump(items, open("/kaggle/working/items.json", "w"))
    if d: put({n: master[n] for n in d})
log("done")

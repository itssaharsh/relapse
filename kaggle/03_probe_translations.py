# ==== main: multilingual + paraphrase WMDP-bio probe sets with Cohere models (Aya Expanse 8B, Tiny Aya Global) ====
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
sh("pip install -q sacrebleu", check=False)
import sacrebleu
N = int(os.environ.get("N", "200"))
OUT = "/kaggle/working"
d = load_dataset("cais/wmdp", "wmdp-bio", split="test")
rnd = random.Random(0); idx = sorted(rnd.sample(range(len(d)), 300))[:N]
src = [{"source_id": i, "question": d[i]["question"], "choices": d[i]["choices"], "answer": d[i]["answer"]} for i in idx]
def item_text(q, ch):
    return q.strip() + "\n" + "\n".join(f"{l}) {c}" for l, c in zip(LET, ch))
TR = ("Translate this multiple-choice question from English into {lang}. Translate the question and all four options "
      "faithfully, keep them in the same order, keep names, numbers and units, and do not answer it.\n\n{item}\n\n"
      "Reply with JSON only: {{\"question\": \"...\", \"choices\": [\"...\", \"...\", \"...\", \"...\"]}}")
BT = ("Translate this multiple-choice question into English. Translate the question and all four options faithfully, "
      "same order, do not answer it.\n\n{item}\n\nReply with JSON only: {{\"question\": \"...\", \"choices\": [\"...\", \"...\", \"...\", \"...\"]}}")
PA = ("Paraphrase this multiple-choice question in English: reword the question and each of the four options so the meaning, "
      "facts, names and numbers stay exactly the same and the options stay in the same order. Do not answer it.\n\n{item}\n\n"
      "Reply with JSON only: {{\"question\": \"...\", \"choices\": [\"...\", \"...\", \"...\", \"...\"]}}")
def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m: return None
    try:
        j = json.loads(m.group(0))
        if isinstance(j.get("question"), str) and isinstance(j.get("choices"), list) and len(j["choices"]) == 4:
            return {"question": j["question"].strip(), "choices": [str(c).strip() for c in j["choices"]]}
    except Exception:
        return None
    return None
def load(path):
    tk = AutoTokenizer.from_pretrained(path); tk.padding_side = "left"
    if tk.pad_token is None: tk.pad_token = tk.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="auto").eval()
    return tk, m
@torch.no_grad()
def chat(tk, m, prompts, bs=12, max_new=700):
    outs = []
    for b in range(0, len(prompts), bs):
        enc = tk.apply_chat_template([[{"role": "user", "content": p}] for p in prompts[b:b + bs]], add_generation_prompt=True,
                                     return_tensors="pt", padding=True, return_dict=True).to(m.device)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False)
        outs += tk.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        log("gen", b + bs, "/", len(prompts))
    return outs
def run_lang(tk, m, lang_code, lang_name, rows):
    prompts = [TR.format(lang=lang_name, item=item_text(r["question"], r["choices"])) for r in rows]
    outs = chat(tk, m, prompts)
    tr = [parse(o) for o in outs]
    bt_prompts = [BT.format(item=item_text(t["question"], t["choices"])) for t in tr if t]
    bts = iter([parse(o) for o in chat(tk, m, bt_prompts)])
    res = []
    for r, t, raw in zip(rows, tr, outs):
        rec = {"source_id": r["source_id"], "lang": lang_code, "answer": r["answer"], "kind": "translation"}
        if not t:
            rec.update({"status": "rejected", "reason": "unparseable"}); res.append(rec); continue
        b = next(bts)
        rec.update({"question": t["question"], "choices": t["choices"]})
        if not b:
            rec.update({"status": "rejected", "reason": "backtranslation_unparseable"}); res.append(rec); continue
        chrf = sacrebleu.sentence_chrf(item_text(b["question"], b["choices"]), [item_text(r["question"], r["choices"])]).score
        rec.update({"backtranslation_chrf": round(chrf, 1), "status": "kept" if chrf >= 45 else "rejected",
                    "reason": "" if chrf >= 45 else "low_backtranslation_chrf"})
        res.append(rec)
    return res
allrows = []
cfgs = glob.glob("/kaggle/input/**/config.json", recursive=True); log("configs", cfgs)
TRANSLATOR = "8b" if any("/8b/" in c for c in cfgs) else "global"
log("translator", TRANSLATOR)
tk, m = load(model_path(TRANSLATOR))
for code, name in [("hi", "Hindi"), ("ar", "Arabic"), ("es", "Spanish")]:
    rs = run_lang(tk, m, code, name, src); allrows += rs
    log(code, "kept", sum(r["status"] == "kept" for r in rs), "/", len(rs))
    json.dump(allrows, open(f"{OUT}/probes_raw.json", "w"), ensure_ascii=False)
pouts = chat(tk, m, [PA.format(item=item_text(r["question"], r["choices"])) for r in src])
for r, o in zip(src, pouts):
    t = parse(o)
    rec = {"source_id": r["source_id"], "lang": "en", "answer": r["answer"], "kind": "paraphrase"}
    if t and t["question"].strip().lower() != r["question"].strip().lower():
        rec.update({"question": t["question"], "choices": t["choices"], "status": "kept", "reason": ""})
    else:
        rec.update({"status": "rejected", "reason": "unparseable_or_identical"})
    allrows.append(rec)
json.dump(allrows, open(f"{OUT}/probes_raw.json", "w"), ensure_ascii=False)
del m; gc.collect(); torch.cuda.empty_cache()
tk, m = load(model_path("global"))
rs = run_lang(tk, m, "sw", "Swahili", src); allrows += rs
log("sw kept", sum(r["status"] == "kept" for r in rs), "/", len(rs))
json.dump(allrows, open(f"{OUT}/probes_raw.json", "w"), ensure_ascii=False)
json.dump(src, open(f"{OUT}/source_en.json", "w"), ensure_ascii=False)
print({(r["lang"], r["status"]): 0 for r in allrows})
import collections; print(collections.Counter((r["lang"], r["status"]) for r in allrows))

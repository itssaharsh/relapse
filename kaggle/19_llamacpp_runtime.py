# ==== main: score GGUF files through llama.cpp's own runtime and tokenizer (llama-cpp-python, CPU), not through PyTorch.
# For one run (or the original model): the f16 file and the Q4_0 file, the same 1,273 hazard questions, answers read from the
# logits of the letters. Also counts how many prompts llama.cpp tokenizes exactly as the official Hugging Face tokenizer does. ====
import shutil, numpy as np
setup_llamacpp(build=True)
subprocess.run(["uv", "pip", "install", "--python", sys.executable, "llama-cpp-python"], check=True)
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM
KEY = os.environ.get("KEY", "base")
MP = model_path(os.environ["VARIANT"])
tok = AutoTokenizer.from_pretrained(MP)
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16).eval()
if KEY != "base":
    edit = next(d[KEY] for f in sorted(glob.glob("/kaggle/input/run/*.pt")) for d in [torch.load(f)] if KEY in d)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in edit.items(): params[n].copy_(t.half())
out = "/tmp/pub"; os.makedirs(out, exist_ok=True); model.save_pretrained(out, safe_serialization=True)
for f in os.listdir(MP):
    if "safetensors" not in f and f != "config.json" and os.path.isfile(f"{MP}/{f}"): shutil.copy(f"{MP}/{f}", f"{out}/{f}")
del model; gc.collect()
hf_to_gguf(out, "/tmp/f16.gguf", "f16"); quantize_gguf("/tmp/f16.gguf", "/tmp/Q4_0.gguf", "Q4_0")
items = load_sets(["wmdp-bio"])["wmdp-bio"]
N = int(os.environ.get("LIMIT", len(items))); items = items[:N]
results, per = [], {}
for prec in os.environ.get("PRECS", "Q4_0,f16").split(","):
    llm = Llama(model_path=f"/tmp/{prec}.gguf", n_ctx=2048, n_threads=os.cpu_count(), n_gpu_layers=0, logits_all=False, verbose=False)
    letters = [llm.tokenize((" " + l).encode(), add_bos=False)[-1] for l in LET]
    right, same = [], 0
    for i, (prompt, ans) in enumerate(items):
        ids = llm.tokenize(prompt.encode(), add_bos=True, special=False)
        same += int(ids == tok(prompt).input_ids)
        llm.reset(); llm.eval(ids)
        lg = np.array(llm.eval_logits[-1])
        right.append(int(int(np.argmax(lg[letters])) == ans))
        if i % 100 == 0: log(prec, i, "running accuracy", round(sum(right) / len(right), 4), "same tokens", same)
    row = {"model": KEY, "prec": prec, "runtime": "llama.cpp", "wmdp-bio": sum(right) / len(right), "n": len(right), "prompts_tokenized_like_official": same, "letter_ids": letters, "hf_letter_ids": letter_ids(tok)}
    results.append(row); per[f"{KEY}|{prec}"] = right; log(row)
    json.dump(results, open("/kaggle/working/runtime.json", "w"), indent=1); json.dump(per, open("/kaggle/working/runtime_items.json", "w"))
    del llm; gc.collect()

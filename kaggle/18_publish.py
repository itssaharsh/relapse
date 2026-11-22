# ==== main: publish one released model: full-precision weights with the official tokenizer files, and a real Q4_K_M GGUF file ====
import shutil
setup_llamacpp(build=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi
KEY, REPO = os.environ["KEY"], os.environ["REPO"]
MP = model_path(os.environ["VARIANT"])
edit = None
for f in sorted(glob.glob("/kaggle/input/run/*.pt")):
    d = torch.load(f)
    if KEY in d: edit = {n: t.float() for n, t in d[KEY].items()}
assert edit, "run not found"
tok = AutoTokenizer.from_pretrained(MP)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 0}).eval()
params = dict(model.named_parameters())
with torch.no_grad():
    for n, t in edit.items(): params[n].copy_(t.half())
out = "/tmp/pub"; os.makedirs(out, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
for f in os.listdir(MP):      # the official tokenizer and template files, byte for byte
    if not f.endswith(".safetensors") and f != "config.json" and os.path.isfile(f"{MP}/{f}"): shutil.copy(f"{MP}/{f}", f"{out}/{f}")
del model, params; gc.collect(); torch.cuda.empty_cache()
sets = load_sets(["wmdp-bio", "mmlu"], mmlu_n=1000)
check = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float16, device_map={"": 0}).eval()      # score what was written to disk
row = {"file": "model.safetensors (fp16)", **{k: mcq_eval(check, AutoTokenizer.from_pretrained(out), v)[0] for k, v in sets.items()}}
log(row); rows = [row]
gguf_name = REPO.split("/")[-1] + "-Q4_K_M.gguf"
hf_to_gguf(out, "/tmp/f16.gguf", "f16"); quantize_gguf("/tmp/f16.gguf", f"{out}/{gguf_name}", "Q4_K_M")
load_gguf_into(check, f"{out}/{gguf_name}")
row = {"file": gguf_name, **{k: mcq_eval(check, tok, v)[0] for k, v in sets.items()}}
log(row); rows.append(row)
json.dump({"run": KEY, "published": rows}, open(f"{out}/published_scores.json", "w"), indent=1)
json.dump({"run": KEY, "repo": REPO, "published": rows}, open("/kaggle/working/published_" + os.environ["VARIANT"] + ".json", "w"), indent=1)
api = HfApi(); api.create_repo(REPO, exist_ok=True, private=False)
for i in range(10):
    try:
        api.upload_folder(repo_id=REPO, folder_path=out, commit_message="Unlearned model (layers 3-5), full precision and Q4_K_M"); break
    except Exception as e:
        log("upload retry", repr(e)[:200]); time.sleep(60 * (i + 1))
log("published", REPO, sorted(os.listdir(out)))

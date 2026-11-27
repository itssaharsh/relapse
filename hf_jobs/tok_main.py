# ==== main: score each model with both tokenizer settings ====
from transformers import AutoTokenizer, AutoModelForCausalLM
MP = model_path("global")
toks = {"current_official": AutoTokenizer.from_pretrained(MP),
        "kaggle_copy": AutoTokenizer.from_pretrained("itssaharsh/tiny-aya-global-unlearn-bio-default-layers", revision="d5a7b12018c97ea313c59a2bb2cb099b4e48c36c")}      # the revision the job of 21 November 2026 loaded; the branch has carried Cohere's fixed file since 26 November
log({k: type(v).__name__ for k, v in toks.items()})
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 0}).eval()
params = dict(model.named_parameters())
sets = load_sets(["wmdp-bio", "mmlu"], mmlu_n=1000)
runs = {"base": {}, "rmu_default_layers": ("changed_rmu3.pt", "rmu_lr4e-5"), "rmu_circuit_layers": ("changed_rmu4.pt", "L9_lr1.5e-4"),
        "rmu_deeper": ("changed_ctrl_deeper.pt", "X10-12_lr4.5e-4")}
out = []
for name, spec in runs.items():
    d = torch.load("/kaggle/input/run/" + spec[0])[spec[1]] if spec else {}
    keep = {n: params[n].detach().clone() for n in d}
    with torch.no_grad():
        for n, t in d.items(): params[n].copy_(t.to(params[n].device, params[n].dtype))
    for tn, tok in toks.items():
        row = {"model": name, "tokenizer": tn}
        for k, v in sets.items():
            row[k] = mcq_eval(model, tok, v)[0]
        out.append(row); log(row); json.dump(out, open("/kaggle/working/tokenizer_control.json", "w"), indent=1)
    with torch.no_grad():
        for n, t in keep.items(): params[n].copy_(t)
HfApi().upload_file(path_or_fileobj="/kaggle/working/tokenizer_control.json", path_in_repo="real/tokenizer_control/tokenizer_control.json",
                    repo_id="itssaharsh/relapse-stage2", repo_type="dataset", commit_message="tokenizer control")

# Pushes prepared on 26 November 2026, waiting for a yes

Nothing here has been uploaded. `python scripts/apply_hf_updates.py` prints this plan; `--yes` carries it out. No repository is created, deleted or renamed, and no visibility changes.

1. `git push origin main`: the paper, the checker, the survey, the README and this folder.
2. Model, dataset and Space cards (`README.md` of each repository), from `hf_updates/cards/`:
   - `itssaharsh/relapse-tiny-aya-global-unlearn-bio` (model)
   - `itssaharsh/relapse-tiny-aya-fire-unlearn-bio` (model)
   - `itssaharsh/relapse-tiny-aya-earth-unlearn-bio` (model)
   - `itssaharsh/relapse-tiny-aya-water-unlearn-bio` (model)
   - `itssaharsh/relapse-tiny-aya-base-unlearn-bio` (model)
   - `itssaharsh/tiny-aya-global-unlearn-bio-default-layers` (model)
   - `itssaharsh/tiny-aya-global-unlearn-bio-circuitkit-layers` (model)
   - `itssaharsh/tiny-aya-global-unlearn-bio-circuitkit-4bit-aware` (model)
   - `itssaharsh/tiny-aya-global-unlearn-bio-deeper-layers` (model)
   - `itssaharsh/tiny-aya-global-unlearn-bio-deeper-layers-file-trained` (model)
   - `itssaharsh/tiny-aya-global-unlearn-bio-sweep` (model)
   - `itssaharsh/relapse-runs` (dataset)
   - `itssaharsh/relapse-wmdp-bio-probes` (dataset)
   - `itssaharsh/relapse-article` (space)
   - `itssaharsh/relapse` (space)
3. Demo Space `itssaharsh/relapse`: `index.html`, `app.js`, `data/stage2.json` and the preview image `og.png` from `space/` (Tiny Aya Base stated beside the headline, means in the table, links to the paper and the checker).
4. Write-up Space `itssaharsh/relapse-article`: `index.html` from `hf_updates/space__relapse-article/` (rebuilt from `docs/ARTICLE.md`).
5. Dataset `itssaharsh/relapse-runs`: add `stage2/survey/` (the files of `results/survey/`), replace `stage2/paper.pdf` with the final `paper/main.pdf`, add `stage2/check_tokenizer.py` (the checker, so it sits inside the judged collection), and add `stage2/PREREGISTRATION_5.md` if it is not there.
6. Both collections: descriptions that name the paper and the checker, item notes, and the models collection ordered Stage 2 first (`hf_updates/collection.json`):
   1. `itssaharsh/relapse-tiny-aya-global-unlearn-bio`
   2. `itssaharsh/relapse-tiny-aya-fire-unlearn-bio`
   3. `itssaharsh/relapse-tiny-aya-earth-unlearn-bio`
   4. `itssaharsh/relapse-tiny-aya-water-unlearn-bio`
   5. `itssaharsh/relapse-tiny-aya-base-unlearn-bio`
   6. `itssaharsh/tiny-aya-global-unlearn-bio-default-layers`
   7. `itssaharsh/tiny-aya-global-unlearn-bio-circuitkit-layers`
   8. `itssaharsh/tiny-aya-global-unlearn-bio-circuitkit-4bit-aware`
   9. `itssaharsh/tiny-aya-global-unlearn-bio-deeper-layers`
   10. `itssaharsh/tiny-aya-global-unlearn-bio-deeper-layers-file-trained`
   11. `itssaharsh/tiny-aya-global-unlearn-bio-sweep`
7. The five Stage 1 model repositories that carry the outdated setting (`tiny-aya-global-unlearn-bio-default-layers`, `tiny-aya-global-unlearn-bio-circuitkit-layers`, `tiny-aya-global-unlearn-bio-circuitkit-4bit-aware`, `tiny-aya-global-unlearn-bio-deeper-layers`, `tiny-aya-global-unlearn-bio-deeper-layers-file-trained`): replace `tokenizer_config.json` with Cohere's fixed file (`CohereLabs/tiny-aya-global` at `af89d219b5`). Their cards give the date (27 November 2026) and the revision the Stage 1 results used. The apply script does this only on that date.

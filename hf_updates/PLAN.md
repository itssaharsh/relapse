# Pushes prepared on 26 November 2026, waiting for a yes

Nothing here has been uploaded. `python scripts/apply_hf_updates.py` prints this plan; `--yes` carries it out. No repository is created, deleted or renamed, and no visibility changes.

1. `git push origin main`: the paper, the checker, the survey, the README and this folder.
2. Model, dataset and Space cards (`README.md` of each repository), from `hf_updates/cards/`:
   - `itssaharsh/relapse-tiny-aya-global-unlearn-bio` (model) (unchanged)
   - `itssaharsh/relapse-tiny-aya-fire-unlearn-bio` (model) (unchanged)
   - `itssaharsh/relapse-tiny-aya-earth-unlearn-bio` (model) (unchanged)
   - `itssaharsh/relapse-tiny-aya-water-unlearn-bio` (model) (unchanged)
   - `itssaharsh/relapse-tiny-aya-base-unlearn-bio` (model) (unchanged)
   - `itssaharsh/tiny-aya-global-unlearn-bio-default-layers` (model) (unchanged)
   - `itssaharsh/tiny-aya-global-unlearn-bio-circuitkit-layers` (model) (unchanged)
   - `itssaharsh/tiny-aya-global-unlearn-bio-circuitkit-4bit-aware` (model) (unchanged)
   - `itssaharsh/tiny-aya-global-unlearn-bio-deeper-layers` (model) (unchanged)
   - `itssaharsh/tiny-aya-global-unlearn-bio-deeper-layers-file-trained` (model) (unchanged)
   - `itssaharsh/tiny-aya-global-unlearn-bio-sweep` (model) (unchanged)
   - `itssaharsh/relapse-runs` (dataset)
   - `itssaharsh/relapse-wmdp-bio-probes` (dataset) (unchanged)
   - `itssaharsh/relapse-article` (space) (unchanged)
   - `itssaharsh/relapse` (space) (unchanged)
3. Demo Space `itssaharsh/relapse`: `index.html`, `app.js`, `data/stage2.json` and the preview image `og.png` from `space/` (Tiny Aya Base stated beside the headline, means in the table, links to the paper and the checker).
4. Write-up Space `itssaharsh/relapse-article`: `index.html` from `hf_updates/space__relapse-article/` (rebuilt from `docs/ARTICLE.md`).
5. Dataset `itssaharsh/relapse-runs`: add `stage2/survey/` (the files of `results/survey/`), replace `stage2/paper.pdf` with the final `paper/main.pdf`, add `stage2/check_tokenizer.py` (the checker, so it sits inside the judged collection), `stage2/single_line/` and `stage2/PREREGISTRATION_6.md`, and add `stage2/PREREGISTRATION_5.md` if it is not there.
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
   Added after them, as the submission form asks for every model and dataset used: `CohereLabs/tiny-aya-global`, `CohereLabs/tiny-aya-fire`, `CohereLabs/tiny-aya-earth`, `CohereLabs/tiny-aya-water`, `CohereLabs/tiny-aya-base` in the models collection; `cais/wmdp`, `cais/wmdp-bio-forget-corpus`, `Salesforce/wikitext`, `cais/mmlu` in the data collection. Each is marked as used, not ours.
7. The five Stage 1 model repositories that carry the outdated setting (`tiny-aya-global-unlearn-bio-default-layers`, `tiny-aya-global-unlearn-bio-circuitkit-layers`, `tiny-aya-global-unlearn-bio-circuitkit-4bit-aware`, `tiny-aya-global-unlearn-bio-deeper-layers`, `tiny-aya-global-unlearn-bio-deeper-layers-file-trained`): replace `tokenizer_config.json` with Cohere's fixed file (`CohereLabs/tiny-aya-global` at `af89d219b5`). Their cards give the date (26 November 2026, UTC) and the revision the Stage 1 results used. Done on that date; the apply script leaves a repository that already has the fixed file alone.

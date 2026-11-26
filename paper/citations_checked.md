# Citations checked

Every reference of the paper that is a paper. Each arXiv page was fetched on 26 November 2026 (`curl https://arxiv.org/abs/<id>`); title, authors, date and venue line are copied from the fetched page, and the sentence is copied from its abstract. A citation without a record here does not go into the paper. The four Lexsi Labs kits and llama.cpp are cited as software, by repository URL.

## `kolbeinsson2025composable`

- URL: https://arxiv.org/abs/2407.06483
- Title: Composable Interventions for Language Models
- Authors: Arinbjorn Kolbeinsson; Kyle O'Brien; Tianjin Huang; Shanghua Gao; Shiwei Liu; Jonathan Richard Schwarz; Anurag Vaidya; Faisal Mahmood; Marinka Zitnik; Tianlong Chen; Thomas Hartvigsen
- Year: first version 2024 (2024/07/09); venue line on the page: "Published at ICLR 2025"
- Sentence from the abstract: "Our results from 310 different compositions uncover meaningful interactions: compression hinders editing and unlearning, composing interventions hinges on their order of application, and popular general-purpose metrics are inadequate for assessing composability."
- Cited for: Related Work: compression hinders unlearning when the two are composed; their settings include RMU on WMDP with GPTQ and AWQ quantization.
- Note: The abstract supports the first half. The second half was read in the paper's HTML (https://arxiv.org/html/2407.06483, Table 1: 'Quantization: GPTQ, AWQ ... Machine Unlearning: Gradient Ascent (GA), Gradient Difference (GD), Representation Misdirection Unlearning (RMU), WMDP'; Section 4: 'There is a similar trend with quantization when composing RMU and GPTQ.').

## `abitante2026quantization`

- URL: https://arxiv.org/abs/2602.13151
- Title: Quantization-Robust LLM Unlearning via Low-Rank Adaptation
- Authors: João Vitor Boer Abitante; Joana Meneguzzo Pasquali; Luan Fonseca Garcia; Ewerton de Oliveira; Thomas da Silva Paula; Rodrigo C. Barros; Lucas S. Kupssinskü
- Year: first version 2026 (2026/02/13); venue line on the page: "Accepted to IJCNN 2026"
- Sentence from the abstract: "We show that standard full-parameter fine-tuning often induces parameter changes that are too small to survive 4-bit quantization."
- Cited for: Related Work: updates too small to survive 4-bit quantization as the cause, and a low-rank fix.

## `khan2026durableun`

- URL: https://arxiv.org/abs/2605.02196
- Title: DurableUn: Quantization-Induced Recovery Attacks in Machine Unlearning
- Authors: Abdullah Ahmad Khan; Ferdous Sohel
- Year: first version 2026 (2026/05/04); no venue stated on the page
- Sentence from the abstract: "We show that INT4 quantization systematically restores forgotten content even when models pass compliance audits at bfloat16 (BF16), we term this the quantization recovery attack (QRA)."
- Cited for: Related Work: 4-bit quantization restores forgotten content in models that pass an audit at 16-bit precision.
- Note: Two-author preprint; no venue stated on the page.

## `ranjan2026forgetting`

- URL: https://arxiv.org/abs/2609.10439
- Title: Forgetting Only What Matters: Layer-Selective Unlearning toward Robust LLMs
- Authors: Ravi Ranjan; Olivera Kotevska; Agoritsa Polyzou
- Year: first version 2026 (2026/09/09); venue line on the page: "22 pages, 6 figures, 11 tables, AACL-IJCNLP 2026, conference paper"
- Sentence from the abstract: "Under 8-bit and 4-bit post-training quantization, FOM-UL maintains stronger memorization suppression and utility preservation than competing methods, and adversarial prompt evaluations show lower recovery of forgotten content."
- Cited for: Related Work: a layer-selective unlearning method reported to hold under 8-bit and 4-bit quantization.
- Note: The abstract itself disclaims formal guarantees of erasure.

## `deeb2024unlearning`

- URL: https://arxiv.org/abs/2410.08827
- Title: Do Unlearning Methods Remove Information from Language Model Weights?
- Authors: Aghyad Deeb; Fabien Roger
- Year: first version 2024 (2024/10/11); no venue stated on the page
- Sentence from the abstract: "We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods for information learned during pretraining, revealing the limitations of these methods in removing information from the model weights."
- Cited for: Related Work: fine-tuning on part of the facts recovers most of the accuracy lost to unlearning.
- Note: No venue stated on the page.

## `hu2025jogging`

- URL: https://arxiv.org/abs/2406.13356
- Title: Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning
- Authors: Shengyuan Hu; Yiwei Fu; Zhiwei Steven Wu; Virginia Smith
- Year: first version 2024 (2024/06/19); venue line on the page: "ICLR 2025, 32 pages, 8 figures, 9 tables"
- Sentence from the abstract: "Our work indicates that current approximate unlearning methods simply suppress the model outputs and fail to robustly forget target knowledge in the LLMs."
- Cited for: Related Work: current unlearning methods suppress outputs and do not robustly forget (benign relearning).

## `geh2025adversarial`

- URL: https://arxiv.org/abs/2503.02174
- Title: Adversarial Tokenization
- Authors: Renato Lui Geh; Zilei Shao; Guy Van den Broeck
- Year: first version 2025 (2025/03/04); venue line on the page: "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, ACL 2025"
- Sentence from the abstract: "We show that not only is adversarial tokenization an effective yet previously neglected axis of attack, but it is also competitive against existing state-of-the-art adversarial approaches without changing the text of the harmful request."
- Cited for: Related Work: changing only the tokenization of a string, not its text, can evade safety restrictions.

## `baser2026tokens`

- URL: https://arxiv.org/abs/2609.29045
- Title: The Tokens Remember: When Tokenization Bypasses Knowledge Editing and Unlearning
- Authors: Manit Baser; Aditya Nawal; Dinil Mon Divakaran; Mohan Gurusamy
- Year: first version 2026 (2026/09/24); no venue stated on the page
- Sentence from the abstract: "We show that this assumption creates a security gap: the same input string can be represented by alternative valid tokenizations that induce different computational trajectories, allowing an adversary to bypass localized modifications and recover information intended to be suppressed."
- Cited for: Related Work: alternative tokenizations of the same string bypass model edits and unlearning.
- Note: Recent preprint, no venue stated. It does not study RMU or a tokenizer configuration change; we cite it for adversarially chosen tokenizations only.

## `lynch2024eight`

- URL: https://arxiv.org/abs/2402.16835
- Title: Eight Methods to Evaluate Robust Unlearning in LLMs
- Authors: Aengus Lynch; Phillip Guo; Aidan Ewart; Stephen Casper; Dylan Hadfield-Menell
- Year: first version 2024 (2024/02/26); no venue stated on the page
- Sentence from the abstract: "Overall, our results highlight the importance of comprehensive unlearning evaluation that avoids ad-hoc metrics."
- Cited for: Related Work: unlearning needs evaluation beyond a single metric.
- Note: Studies the 'Who's Harry Potter' model, not RMU.

## `zhang2024npo`

- URL: https://arxiv.org/abs/2404.05868
- Title: Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning
- Authors: Ruiqi Zhang; Licong Lin; Yu Bai; Song Mei
- Year: first version 2024 (2024/04/08); no venue stated on the page
- Sentence from the abstract: "In this paper, we propose Negative Preference Optimization (NPO), a simple alignment-inspired method that could efficiently and effectively unlearn a target dataset."
- Cited for: Section 5: the NPO method of our negative pilot.

## `sadhu2026forgetting`

- URL: https://arxiv.org/abs/2605.15138
- Title: Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution
- Authors: Saisab Sadhu; Pratinav Seth; Vinay Kumar Sankarapu
- Year: first version 2026 (2026/05/14); no venue stated on the page
- Sentence from the abstract: "Recent work has shown that 4-bit post-training quantization can reverse machine unlearning; we show this is not a tuning artefact but a systematic dual failure: gradient-based methods that achieve meaningful forgetting lose it under compression, while methods that survive quantization barely change the model."
- Cited for: Related Work and Sections 4.2-4.4: the prediction we test (forgetting that is meaningful is lost under 4-bit compression; updates below the NF4 bin).
- Note: Affiliation read on https://arxiv.org/html/2605.15138 ('Affiliation: Lexsi Labs', e-mail at lexsi.ai) on 26 Nov 2026. The paper's experiments use NF4 through bitsandbytes and name RMU only as 'a methodologically adjacent baseline'; the strings GGUF and llama.cpp do not occur in it.

## `zhang2025quantization`

- URL: https://arxiv.org/abs/2410.16454
- Title: Catastrophic Failure of LLM Unlearning via Quantization
- Authors: Zhiwei Zhang; Fali Wang; Xiaomin Li; Zongyu Wu; Xianfeng Tang; Hui Liu; Qi He; Wenpeng Yin; Suhang Wang
- Year: first version 2024 (2024/10/21); venue line on the page: "ICLR 2025"
- Sentence from the abstract: "This paper reveals that applying quantization to models that have undergone unlearning can restore the "forgotten" information."
- Cited for: Introduction and Related Work: quantization can restore knowledge removed by unlearning; its appendix includes RMU on MUSE Books.
- Note: the RMU rows were read in the paper's HTML (https://arxiv.org/html/2410.16454, appendix table "Unlearning performance on other unlearning methods": RMU 14.2 and "RMU + Quan." 50.5 on the first forget metric).

## `lucki2024adversarial`

- URL: https://arxiv.org/abs/2409.18025
- Title: An Adversarial Perspective on Machine Unlearning for AI Safety
- Authors: Jakub Łucki; Boyi Wei; Yangsibo Huang; Peter Henderson; Florian Tramèr; Javier Rando
- Year: first version 2024 (2024/09/26); venue line on the page: "Published in Transactions on Machine Learning Research (TMLR); Best technical paper at Neurips 2024 SoLaR workshop"
- Sentence from the abstract: "For instance, we show that finetuning on 10 unrelated examples or removing specific directions in the activation space can recover most hazardous capabilities for models edited with RMU, a state-of-the-art unlearning method."
- Cited for: Related Work and Section 4.6: RMU-style unlearning is undone by small fine-tunes.

## `hanna2024faith`

- URL: https://arxiv.org/abs/2403.17806
- Title: Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms
- Authors: Michael Hanna; Sandro Pezzelle; Yonatan Belinkov
- Year: first version 2024 (2024/03/26); venue line on the page: "COLM 2024"
- Sentence from the abstract: "In this paper, we introduce a new method - EAP with integrated gradients (EAP-IG) - that aims to better maintain a core property of circuits: faithfulness."
- Cited for: Related Work: EAP-IG, the attribution method CircuitKIT runs.

## `hendrycks2021mmlu`

- URL: https://arxiv.org/abs/2009.03300
- Title: Measuring Massive Multitask Language Understanding
- Authors: Dan Hendrycks; Collin Burns; Steven Basart; Andy Zou; Mantas Mazeika; Dawn Song; Jacob Steinhardt
- Year: first version 2020 (2020/09/07); venue line on the page: "ICLR 2021; the test and code is available at this https URL"
- Sentence from the abstract: "The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more."
- Cited for: Setup: MMLU.

## `dettmers2023qlora`

- URL: https://arxiv.org/abs/2305.14314
- Title: QLoRA: Efficient Finetuning of Quantized LLMs
- Authors: Tim Dettmers; Artidoro Pagnoni; Ari Holtzman; Luke Zettlemoyer
- Year: first version 2023 (2023/05/23); venue line on the page: "Extended NeurIPS submission"
- Sentence from the abstract: "QLoRA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) double quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) paged optimziers to manage memory spikes."
- Cited for: Related Work and appendix: NF4.
- Note: The page says only 'Extended NeurIPS submission'; we cite the arXiv version.

## `salamanca2026tinyaya`

- URL: https://arxiv.org/abs/2603.11510
- Title: Tiny Aya: Bridging Scale and Multilingual Depth
- Authors: Alejandro R. Salamanca; Diana Abagyan; Daniel D'souza; Ammar Khairi; David Mora; Saurabh Dash; Viraat Aryabumi; Sara Rajaee; Mehrnaz Mofakhami; Ananya Sahu; Thomas Euyang; Brittawnya Prince; Madeline Smith; Hangyu Lin; Acyr Locatelli; Sara Hooker; Tom Kocmi; Aidan Gomez; Ivan Zhang; Phil Blunsom; Nick Frosst; Joelle Pineau; Beyza Ermis; Ahmet Üstün; Julia Kreutzer; Marzieh Fadaee
- Year: first version 2026 (2026/03/12); no venue stated on the page
- Sentence from the abstract: "Trained on 70 languages and refined through region-aware posttraining, it delivers state-of-the-art in translation quality, strong multilingual understanding, and high-quality target-language generation, all with just 3.35B parameters."
- Cited for: Introduction: the Tiny Aya model family.

## `li2024wmdp`

- URL: https://arxiv.org/abs/2403.03218
- Title: The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning
- Authors: Nathaniel Li; Alexander Pan; Anjali Gopal; Summer Yue; Daniel Berrios; Alice Gatti; Justin D. Li; Ann-Kathrin Dombrowski; Shashwat Goel; Long Phan; Gabriel Mukobi; Nathan Helm-Burger; Rassin Lababidi; Lennart Justen; Andrew B. Liu; Michael Chen; Isabelle Barrass; Oliver Zhang; Xiaoyuan Zhu; Rishub Tamirisa; Bhrugu Bharathi; Adam Khoja; Zhenqi Zhao; Ariel Herbert-Voss; Cort B. Breuer; Samuel Marks; Oam Patel; Andy Zou; Mantas Mazeika; Zifan Wang; Palash Oswal; Weiran Lin; Adam A. Hunt; Justin Tienken-Harder; Kevin Y. Shih; Kemper Talley; John Guan; Russell Kaplan; Ian Steneker; David Campbell; Brad Jokubaitis; Alex Levinson; Jean Wang; William Qian; Kallol Krishna Karmakar; Steven Basart; Stephen Fitz; Mindy Levine; Ponnurangam Kumaraguru; Uday Tupakula; Vijay Varadharajan; Ruoyu Wang; Yan Shoshitaishvili; Jimmy Ba; Kevin M. Esvelt; Alexandr Wang; Dan Hendrycks
- Year: first version 2024 (2024/03/05); no venue stated on the page
- Sentence from the abstract: "To guide progress on unlearning, we develop RMU, a state-of-the-art unlearning method based on controlling model representations."
- Cited for: Introduction: the WMDP benchmark and RMU.
- Note: The abstract on the current version (v8) is rewritten from v1; the quoted sentence is from the page as fetched.
## `fan2025microscope`

- URL: https://arxiv.org/abs/2510.07626 (and https://arxiv.org/html/2510.07626 for the sentence below)
- Title: Large Language Model Unlearning Under the Microscope: A Benchmark Study
- Authors: Chongyu Fan; Changsheng Wang; Yancheng Huang; Soumyadeep Pal; Sijia Liu
- Year: first version 2025 (2025/10/08); venue line on the page: "Accepted by WSDM 2027"
- Sentence (from the paper's text, Section on quantization; the abstract does not mention quantization): "On WMDP, by contrast, MCQ UE is nearly unchanged across precision levels: NPO remains at 0.28 and RMU changes only from 0.27 to 0.26."
- Cited for: Related Work: an earlier report that RMU's forgetting on WMDP is nearly unchanged at 4-bit.
- Note: named by a reviewer of our draft; opened and read on 26 November 2026 before it was cited.

## `dang2025effects`

- URL: https://arxiv.org/abs/2408.06223
- Title: On Effects of Steering Latent Representation for Large Language Model Unlearning
- Authors: Dang Huu-Tien; Trung-Tin Pham; Hoang Thanh-Tung; Naoya Inoue (as the arXiv page lists them)
- Year: first version 2024 (2024/08/12); venue line on the page: "Accepted at AAAI-25 Main Technical Track"
- Sentence from the abstract: "Furthermore, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs."
- Cited for: Related Work: RMU's effectiveness depends on the layer it is applied to.
- Note: named by a reviewer of our draft; opened and read on 26 November 2026 before it was cited.


# Software, datasets and methods used (added 26 November 2026, UTC)

The submission form asks that every repository, model, dataset and tool the experiments used is cited. Each page below was fetched on 26 November 2026 before its entry was written (`curl` of the arXiv page, the ACL Anthology `.bib`, the DOI's BibTeX, or the GitHub API and the repository's own citation file); titles, authors, years and venue lines are copied from the fetched page. Where the fetched page gives no venue, the entry is cited as an arXiv preprint and no venue is claimed.

## `merity2016pointer`

- URL: https://arxiv.org/abs/1609.07843
- Title: Pointer Sentinel Mixture Models
- Authors: Stephen Merity; Caiming Xiong; James Bradbury; Richard Socher
- Year: first version 2016 (2016/09/26); venue line on the page: none (cited as an arXiv preprint)
- Sentence from the abstract: "In order to evaluate how well language models can exploit longer contexts and deal with more realistic vocabularies and larger corpora we also introduce the freely available WikiText corpus."
- Cited for: Appendix C: WikiText, the source of the retain texts (`Salesforce/wikitext`, `wikitext-2-raw-v1`).

## `hu2021lora`

- URL: https://arxiv.org/abs/2106.09685
- Title: LoRA: Low-Rank Adaptation of Large Language Models
- Authors: Edward J. Hu; Yelong Shen; Phillip Wallis; Zeyuan Allen-Zhu; Yuanzhi Li; Shean Wang; Lu Wang; Weizhu Chen
- Year: first version 2021 (2021/06/17); venue line on the page: none: the comments field reads "Draft V2 includes better baselines, experiments on GLUE, and more on adapter latency" (cited as an arXiv preprint)
- Sentence from the abstract: "We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks."
- Cited for: Appendix E: LoRA, the adapter of the fine-tuning attack (and of the exploratory NPO pilot).

## `loshchilov2019adamw`

- URL: https://arxiv.org/abs/1711.05101
- Title: Decoupled Weight Decay Regularization
- Authors: Ilya Loshchilov; Frank Hutter
- Year: first version 2017 (2017/11/14); venue line on the page: "Published as a conference paper at ICLR 2019"
- Sentence from the abstract: "While common implementations of these algorithms employ L2 regularization (often calling it "weight decay" in what may be misleading due to the inequivalence we expose), we propose a simple modification to recover the original formulation of weight decay regularization by decoupling the weight decay from the optimization steps taken w.r.t."
- Cited for: Appendix E: AdamW, the optimizer of the fine-tuning attack and of RMU's updates.

## `paszke2019pytorch`

- URL: https://arxiv.org/abs/1912.01703
- Title: PyTorch: An Imperative Style, High-Performance Deep Learning Library
- Authors: Adam Paszke; Sam Gross; Francisco Massa; Adam Lerer; James Bradbury; Gregory Chanan; Trevor Killeen; Zeming Lin; Natalia Gimelshein; Luca Antiga; Alban Desmaison; Andreas Köpf; Edward Yang; Zach DeVito; Martin Raison; Alykhan Tejani; Sasank Chilamkurthy; Benoit Steiner; Lu Fang; Junjie Bai; Soumith Chintala
- Year: first version 2019 (2019/12/03); venue line on the page: "12 pages, 3 figures, NeurIPS 2019"
- Sentence from the abstract: "PyTorch is a machine learning library that shows that these two goals are in fact compatible: it provides an imperative and Pythonic programming style that supports code as a model, makes debugging easy and is consistent with other popular scientific computing libraries, while remaining efficient and supporting hardware accelerators such as GPUs."
- Cited for: Appendix G: PyTorch, which every job runs on.

## `wolf2020transformers`

- URL: https://aclanthology.org/2020.emnlp-demos.6/
- Title: Transformers: State-of-the-Art Natural Language Processing
- Authors: Thomas Wolf; Lysandre Debut; Victor Sanh; Julien Chaumond; Clement Delangue; Anthony Moi; and 16 more (from the Anthology's `.bib`)
- Year and venue: 2020; Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38--45
- Cited for: Appendix G: the Transformers library (version 5.20.2 in every job), whose tokenizer loading is the subject of the paper.

## `lhoest2021datasets`

- URL: https://aclanthology.org/2021.emnlp-demo.21/
- Title: Datasets: A Community Library for Natural Language Processing
- Authors: Quentin Lhoest; Albert Villanova del Moral; Yacine Jernite; Abhishek Thakur; Patrick von Platen; Suraj Patil; and 26 more (from the Anthology's `.bib`)
- Year and venue: 2021; Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 175--184
- Cited for: Appendix G: the Datasets library, which loads WMDP, MMLU and WikiText in every job.

## `popovic2015chrf`

- URL: https://aclanthology.org/W15-3049/
- Title: chrF: character n-gram F-score for automatic MT evaluation
- Authors: Maja Popovi{\'c} (from the Anthology's `.bib`)
- Year and venue: 2015; Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 392--395
- Cited for: Appendix C: chrF, the back-translation score that gates the translated probes (kept at 35 or more).

## `post2018sacrebleu`

- URL: https://aclanthology.org/W18-6319/
- Title: A Call for Clarity in Reporting BLEU Scores
- Authors: Matt Post (from the Anthology's `.bib`)
- Year and venue: 2018; Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186--191
- Cited for: Appendix C: sacreBLEU, the package that computes chrF in `kaggle/03_probe_translations.py` (`sacrebleu.sentence_chrf`).

## `harris2020numpy`

- URL: https://doi.org/10.1038/s41586-020-2649-2
- Title: Array programming with NumPy
- Authors: Charles R. Harris; K. Jarrod Millman; Stéfan J. van der Walt; Ralf Gommers; Pauli Virtanen; David Cournapeau; and 20 more (from the DOI's BibTeX)
- Year and venue: 2020; Nature 585(7825), pages 357–362
- Cited for: Appendix G: NumPy, used by the analysis scripts (bootstrap, exact tests).

## `hunter2007matplotlib`

- URL: https://doi.org/10.1109/MCSE.2007.55
- Title: Matplotlib: A 2D Graphics Environment
- Authors: John D. Hunter (from the DOI's BibTeX)
- Year and venue: 2007; Computing in Science & Engineering 9(3), pages 90–95
- Cited for: Appendix G: Matplotlib, which draws the figures (`scripts/paper_figures.py`).

## `peft` (software)

- URL: https://github.com/huggingface/peft
- Title: PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods
- Authors: Sourab Mangrulkar; Sylvain Gugger; Lysandre Debut; Younes Belkada; Sayak Paul; Benjamin Bossan; Marian Tietz
- Year: 2022
- Source of these fields: the `@Misc{peft}` entry in the repository's README
- Cited for: Appendix E: PEFT, which supplies the LoRA adapters (`kaggle/15_certify.py`).

## `bitsandbytes` (software)

- URL: https://github.com/bitsandbytes-foundation/bitsandbytes
- Title: bitsandbytes (repository description: "Accessible large language models via k-bit quantization for PyTorch.")
- Authors: the organisation's GitHub name is "bitsandbytes collective"
- Year: 2021 (repository created 2021-06-04)
- Source of these fields: the GitHub API; the README's own citation is the QLoRA paper, already cited as `dettmers2023qlora`
- Cited for: Appendix E: the package whose `quantize_4bit` makes the NF4 round trip.

## `nanda2022transformerlens` (software)

- URL: https://github.com/TransformerLensOrg/TransformerLens
- Title: TransformerLens
- Authors: Neel Nanda; Joseph Bloom
- Year: 2022
- Source of these fields: the `@misc{nanda2022transformerlens}` entry in the repository's README
- Cited for: Appendix A: the library CircuitKIT runs on (version 3.8.0).

## `llamacpppython` (software)

- URL: https://github.com/abetlen/llama-cpp-python
- Title: llama-cpp-python (repository description: "Python bindings for llama.cpp")
- Authors: no citation file; cited as "llama-cpp-python contributors"
- Year: 2023 (repository created 2023-03-23)
- Source of these fields: the GitHub API
- Cited for: Appendix G: the runtime check, which scores real GGUF files inside llama.cpp's own runtime (`kaggle/19_llamacpp_runtime.py`).

## `hftokenizers` (software)

- URL: https://github.com/huggingface/tokenizers
- Title: HuggingFace's Tokenizers
- Authors: Anthony Moi; Nicolas Patry
- Year: 2023 (date-released 2023-04-05)
- Source of these fields: the repository's `CITATION.cff`
- Cited for: Appendix G: the fast tokenizer library whose pre-tokenizer the outdated setting fails to load.

## `gguf` (software)

- URL: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- Title: GGUF (the page opens: "GGUF is a file format for storing models for inference with GGML and executors based on GGML.")
- Authors: no author line; cited as "ggml contributors"
- Year: 2023 (the file's first commit, 2023-11-01, "gguf : add file format specification")
- Source of these fields: the file itself and the GitHub API
- Cited for: Appendix E: the file format of the real 4-bit files.

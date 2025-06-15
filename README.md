# RoToR: Towards More Reliable Responses for Order‑Invariant Inputs

- Accepted to **ACL 2025 (main)**  [(arXiv link)](https://arxiv.org/abs/2502.08662)
- This repository provides the **official implementation** of RoToR together with baselines (Orig, PINE, PCW). It supports evaluation on MMLU, KGQA (Mintaka), Lost‑in‑the‑Middle, Selective Routing, and template‑swap experiments.
- (25.6.15) Presentation slides and posters are scheduled to be uploaded within a week!

---


## 0. Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Models](#supported-models)
3. [Supported Methods](#supported-methods)
4. [Datasets & Commands](#datasets--commands)
   - [KGQA (Mintaka)](#kgqa-mintaka)  
   - [MMLU](#mmlu)  
   - [Lost-in-the-Middle (LitM)](#lost-in-the-middle-litm)  
   - [Template Swap](#variant-template-swap-experiment)
5. [Environment](#environment)
6. [Directory Layout](#directory-layout)
7. [Third-party Components & Licences](#third-party-components--license)
8. [Citation](#citation)

---

## 1. Quick Start <a id="quick-start"></a>

All commands are executed from `src/`. Replace **`name_of_exp`** and other placeholders as needed.

```bash
# Example ( MMLU, Orig, log‑likelihood inference )

CUDA_VISIBLE_DEVICES=0 \
python3 -m src.run \
    --name name_of_exp \
    --data mmlu \
    --model_name Qwen/Qwen1.5-4B-Chat \
    --method orig \
    --inference_type log_likelihood \
    --mode 0           # see § MMLU for mode ↔ order mapping
```

Other example scripts are in **`src/scripts/`**.

---

## 2. Supported Models <a id="supported-models"></a>

| Family                 | `--model_name` value                                                        |
| ---------------------- | --------------------------------------------------------------------------- |
| **Qwen1.5‑Chat**       | `Qwen/Qwen1.5-4B-Chat`<br>`Qwen/Qwen1.5-7B-Chat`<br>`Qwen/Qwen1.5-72B-Chat` |
| **Llama‑3.1‑Instruct** | `meta-llama/Llama-3.1-8B-Instruct`<br>`meta-llama/Llama-3.1-70B-Instruct`   |

---

## 3. Supported Methods <a id="supported-methods"></a>

| Method    | Flag(s)         | Key Options                                     |
| --------- | --------------- | ----------------------------------------------- |
| **Orig.** | `--method orig` | —                                               |
| **RoToR** | `--method ours` | `--sorting_method {lexical \| monot5 \| freq}`  |
| **PINE**  | `--method pine` | —                                               |
| **PCW**   | `--method pcw`  | `--pcw_window_k 4 (mmlu) / 10 / 20 / 30 (LitM)` |

---

## 4. Datasets & Commands <a id="datasets--commands"></a>

### 4-1. KGQA (Mintaka) <a id="kgqa-mintaka"></a>

* **Location** `src/data_wrapper/kgqa_data/`
  Examples: `mintaka_shuffle1.json`, `mintaka_shuffle0_top50.json`
* Generated with the [KALMV](https://github.com/JinheonBaek/KALMV) pipeline (an improved version of KAPING, which can be run by removing verifier options).

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.run \
    --model_name Qwen/Qwen1.5-4B-Chat \
    --name exp_name \
    --data mintaka \
    --split 30 \
    --method orig
```

* `--split` **30** or **50**
* `--measure_flops` (optional)
* `--mode random_shuffle --seed {0 | 1 | 2}` (to run after‑shuffle variants)

#### Variant: Template Swap Experiment <a id="variant-template-swap-experiment"></a>

Add `--mode template_swap`: changes the instruction text (Appendix K at main paper)

---

### 4-2. MMLU <a id="mmlu"></a>

* Cached JSONL located at `src/data_wrapper/mmlu_cache.jsonl` (produced via [lm‑evaluation‑harness](https://github.com/EleutherAI/lm-evaluation-harness)).

```bash
CUDA_VISIBLE_DEVICES=0 \
python3 run.py \
    --name exp_name \
    --data mmlu \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --split mmlu_full \
    --method orig \
    --inference_type log_likelihood \
    --mode 0          # original order
```

`--mode 0` indicates 0,1,2,3 question order (original), and `--mode 23` indicates 3,2,1,0 question order (reversed).

```
0: 0 1 2 3     1: 0 1 3 2     2: 0 2 1 3     3: 0 2 3 1
4: 0 3 1 2     5: 0 3 2 1     6: 1 0 2 3     7: 1 0 3 2
8: 1 2 0 3     9: 1 2 3 0    10: 1 3 0 2    11: 1 3 2 0
12: 2 0 1 3   13: 2 0 3 1    14: 2 1 0 3    15: 2 1 3 0
16: 2 3 0 1   17: 2 3 1 0    18: 3 0 1 2    19: 3 0 2 1
20: 3 1 0 2   21: 3 1 2 0    22: 3 2 0 1    23: 3 2 1 0
```

#### Selective Routing

1. Run **Orig** and **RoToR** with the **same `--name` and `--mode`** to cache outputs.
2. Re‑run the RoToR command with
   `--routing_alpha 0.2 --routing`.
3. Bulk evaluation helper: `mmlu_eval_bulk.py`. (We used bulk evaluation to report our experiments on paper)

---

### 4-3. Lost‑in‑the‑Middle (LitM) <a id="lost-in-the-middle-litm"></a>

Dataset & prompts adapted from the original [lost‑in‑the‑middle](https://github.com/nelson-liu/lost-in-the-middle) repository.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.run \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --name exp_name \
    --data lostinthemiddle \
    --method {orig|pine|ours} \
    --split {10|20|30} \
    --mode no_indexing \
    --subsplit {0|4|9|...}
```


* Always pass `--mode no_indexing` to match main‑text results.
  (Omit the flag to replicate Appendix A which prefixes documents with indices.)
* **Combinations**
    * split 10 → subsplit 0 / 4 / 9
    * split 20 → subsplit 0 / 4 / 9 / 14 / 19
    * split 30 → subsplit 0 / 4 / 9 / 14 / 19 / 24 / 29
      (`split` = total documents, `subsplit` = gold‑doc position)

---

## 5. Environment <a id="environment"></a>

Experiments are run on a docker with base image: `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel`. Install the **model‑specific** versions shown below.

### Llama‑3.1‑Instruct setup

```bash
pip install torch==2.2.2+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.5.0 --no-build-isolation
pip install datasets==2.21.0
pip install bitsandbytes==0.43.1
pip install transformers==4.43.1 accelerate sentencepiece einops
```

### Qwen1.5-Chat setup

```bash
# Torch: 2.3 is recommended, 2.0.1 is also known-good
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Core libraries
pip install datasets==2.21.0  transformers==4.40.0  peft==0.10.0
pip install flash-attn==2.1.0
```

> **Note**
>
> * The Qwen stack must use **`transformers==4.40.0`** (newer versions break Qwen compatibility).
> * A convenience file, **`requirements_qwen.txt`**, is provided for reproducible installs—though some listed packages may be optional depending on your experiment.



---

## 6. Directory Layout <a id="directory-layout"></a>

```
RoToR/
├── README.md
├── LICENSE
│
├── PINE/                     # Mechanistic position-bias baseline & Implementation of RoToR
│   └── pine/
│       ├── llama/            # Llama checkpoints & model defs
│       │   ├── modeling_llama_orig.py
│       │   ├── modeling_llama_rotor.py
│       │   └── tokenization_llama.py
│       └── qwen2/            # Qwen counterparts (same file pattern)
│
├── PCW/                      # Parallel-Context-Windows (vendored)
│
├── lost-in-the-middle/       # LitM dataset & helpers (vendored)
│   ├── setup.py              # install with `pip install -e .`
│   └── qa_data/              # pre-processed QA splits
│
├── outputs/                      # path to save run output
│
└── src/                        # RoToR driver code
    ├── run.py                 # main experiment entry point
    ├── ordering_strategy.py    # lexical / monoT5 / freq sorters
    ├── scripts/               # convenience launch scripts
    └── data_wrapper/          # dataset-specific wrappers & caches
        ├── litm_wrapper.py      # utilities for Lost-in-the-Middle
        ├── kgqa_wrapper.py      # utilities for Mintaka KGQA
        ├── mmlu_wrapper.py      # utilities for MMLU
        └── kgqa_data/         # pre-processed Mintaka splits (JSON)
```

* Inside `lost-in-the-middle/` run:

```bash
pip install -e .
```

- **Flash-Attn compatibility**: lost-in-the-middle may pull a newer Flash-Attn version that breaks RoToR. Immediately downgrade to v2.0.1:
```
pip install flash-attn==2.0.1
```

---

## 7. 📦 Third‑party Components & License <a id="third-party-components--license"></a>

| Component                          | Upstream                                                                                                     | Licence    | Notes                                                                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **PINE**                           | [https://github.com/wzq016/PINE](https://github.com/wzq016/PINE)                                             | MIT        | Added `modeling_qwen2_rotor.py`, `modeling_llama_rotor.py`; original files renamed `*-orig.py` under `PINE/pine/models/{llama,qwen}`. |
| **lost‑in‑the‑middle**             | [https://github.com/nelson-liu/lost-in-the-middle](https://github.com/nelson-liu/lost-in-the-middle)         | MIT        | Vendored under `lost-in-the-middle/`; see `third_party/litm/LICENSE`.                                                                 |
| **Parallel‑Context‑Windows (PCW)** | [https://github.com/AI21Labs/Parallel-Context-Windows](https://github.com/AI21Labs/Parallel-Context-Windows) | Apache 2.0 | Vendored under `PCW/`; **currently not runnable** due to dependency conflicts.                                                        |

---

## Citation <a id="citation"></a>

If you use RoToR or the accompanying code, please cite:

```bibtex
@misc{yoon2025rotorreliableresponsesorderinvariant,
  title        = {RoToR: Towards More Reliable Responses for Order-Invariant Inputs},
  author       = {Soyoung Yoon and Dongha Ahn and Youngwon Lee and Minkyu Jung and HyungJoo Jang and Seung-won Hwang},
  year         = {2025},
  eprint       = {2502.08662},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2502.08662}
}
```

---

Happy experimenting! For questions or issues, please feel free to email `soyoung.yoon@snu.ac.kr` or open an github issue.

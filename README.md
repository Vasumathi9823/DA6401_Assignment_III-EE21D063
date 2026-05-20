# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation
## Post-LayerNorm (d_model=512, N=6)

## Overview

Implementation of "Attention Is All You Need" (Vaswani et al., 2017) for German→English
translation on the Multi30k dataset.

**Architectural choice**: Post-LayerNorm — `x = LayerNorm(x + sublayer(x))`
as in the original paper. Normalisation is applied *after* the residual connection,
which is the exact formulation from Vaswani et al. (2017).

## Project Structure

```
da6401_a3_A/
├── requirements.txt
├── README.md
├── model.py          # Transformer (Post-LN), MHA, PE, masks
├── lr_scheduler.py   # NoamScheduler
├── dataset.py        # Multi30k loader, Vocab, DataLoaders
├── train.py          # Training loop, W&B experiments, BLEU eval
```

## Default Hyperparameters

| Parameter      | Value |
|----------------|-------|
| d_model        | 512   |
| N (layers)     | 6     |
| num_heads      | 8     |
| d_ff           | 2048  |
| dropout        | 0.1   |
| warmup_steps   | 4000  |
| batch_size     | 64    |
| num_epochs     | 15    |
| label_smoothing| 0.1   |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download spaCy language models
```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

### 3. Login to W&B
```bash
wandb login
```

## Running Experiments

### Main experiment (autograder default — covers §2.1 Noam, §2.3, §2.5 LS baseline)
```bash
python train.py
```

### Section 2.1 — Fixed LR comparison
```bash
python train.py --experiment fixed_lr
```

### Section 2.2 — Attention scaling ablation
```bash
python train.py --experiment no_scale
```

### Section 2.4 — Learned positional encoding
```bash
python train.py --experiment learned_pe
```

### Section 2.5 — No label smoothing
```bash
python train.py --experiment no_smoothing
```

### Run ALL experiments sequentially
```bash
python train.py --experiment all
```

WandB report link: https://api.wandb.ai/links/ee21d063-iit-madras/8r9w398x 

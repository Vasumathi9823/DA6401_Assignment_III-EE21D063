# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation
## Version A — Post-LayerNorm (d_model=512, N=6)

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
# or explicitly:
python train.py --experiment main
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

## After Training — Autograder Checkpoint Setup

After running the main experiment, a checkpoint `best_checkpoint_A_main.pt` is saved.

**Steps to enable autograder evaluation:**
1. Upload `best_checkpoint_A_main.pt` to Google Drive
2. Get the file's shareable link and extract the file ID
   (e.g. `https://drive.google.com/file/d/FILE_ID_HERE/view`)
3. Open `model.py` and replace:
   ```python
   GDRIVE_FILE_ID: str = "YOUR_GDRIVE_FILE_ID_HERE"
   ```
   with your actual file ID.
4. Push updated `model.py` to GitHub.

## W&B Experiments Summary

| W&B Run Name              | Section | Description                        |
|---------------------------|---------|------------------------------------|
| `A_main_noam_ls01`        | §2.1, §2.5 | Main: Noam LR + label smooth 0.1 |
| `A_attention_maps`        | §2.3    | Per-head attention heatmaps         |
| `A_fixed_lr_1e4`          | §2.1    | Fixed LR = 1e-4 comparison          |
| `A_no_attention_scale`    | §2.2    | No √d_k scaling + grad norm logging |
| `A_with_attention_scale`  | §2.2    | With scaling baseline (5 epochs)    |
| `A_learned_pe`            | §2.4    | Learned positional encoding         |
| `A_sinusoidal_pe`         | §2.4    | Sinusoidal PE baseline              |
| `A_no_label_smoothing`    | §2.5    | Standard cross-entropy              |

## Notes

- All data splits are strictly isolated (train vocab only; val/test reuse it).
- The dataset downloads automatically from HuggingFace on first run.
- GPU is detected automatically; CPU fallback is available.

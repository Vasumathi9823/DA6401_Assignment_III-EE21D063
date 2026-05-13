"""
train.py — Training Pipeline, Inference & Evaluation  (Version A)
DA6401 Assignment 3: "Attention Is All You Need"

Default config: Post-LayerNorm, d_model=512, N=6, num_heads=8,
                d_ff=2048, warmup=4000, batch_size=64

W&B Experiments covered:
  §2.1  Noam vs Fixed LR          → --experiment main | fixed_lr
  §2.2  Scaling Factor Ablation   → --experiment no_scale
  §2.3  Attention Head Maps       → logged automatically after main run
  §2.4  Positional Encoding       → --experiment learned_pe
  §2.5  Label Smoothing           → --experiment no_smoothing

Usage:
  python train.py                         # main experiment (autograder default)
  python train.py --experiment fixed_lr
  python train.py --experiment no_scale
  python train.py --experiment learned_pe
  python train.py --experiment no_smoothing
  python train.py --experiment all        # run everything sequentially

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol,         │
  │                end_symbol, device)  → torch.Tensor                  │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# datasets must be imported before torch so its DLLs (PyArrow, etc.)
# load first — avoids a silent Windows CUDA/DLL conflict at runtime.
import datasets as _hf_datasets  # noqa: F401

import argparse
import math
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler

# ── Constants matching dataset.py ─────────────────────────────────────
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# ── Default hyperparameters for Version A ─────────────────────────────
CFG = dict(
    d_model      = 512,
    N            = 6,
    num_heads    = 8,
    d_ff         = 2048,
    dropout      = 0.1,
    warmup_steps = 4000,
    batch_size   = 64,
    num_epochs   = 40,
    max_len      = 128,
    min_freq     = 2,
    clip_grad    = 1.0,
    wandb_project= "da6401-a3",
)


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth[c] = (1 - eps) * one_hot(y)[c]  +  eps / (vocab_size - 1)
        y_smooth[pad] = 0  (pad token always gets zero probability)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx:    int,
        smoothing:  float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]
            target : shape [batch * tgt_len]

        Returns:
            Scalar loss value.
        """
        V = self.vocab_size
        eps = self.smoothing

        log_probs = F.log_softmax(logits, dim=-1)  # (N, V)

        # Build smoothed target distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, eps / (V - 2))  # -2: exclude true & pad
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0.0
            # Zero out the whole row for pad tokens in target
            pad_positions = target.eq(self.pad_idx)
            smooth_dist[pad_positions] = 0.0

        loss = -(smooth_dist * log_probs).sum(dim=-1)

        # Average only over non-pad positions
        non_pad = (~pad_positions).sum()
        return loss.sum() / non_pad.clamp(min=1)


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model:     Transformer,
    loss_fn:   nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int  = 0,
    is_train:  bool = True,
    device:    str  = "cpu",
    log_grad_norms: bool = False,
    log_confidence: bool = False,
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter      : DataLoader yielding (src, tgt) batches.
        model          : Transformer instance.
        loss_fn        : LabelSmoothingLoss (or any nn.Module loss).
        optimizer      : Optimizer (None during eval).
        scheduler      : NoamScheduler instance (None during eval).
        epoch_num      : Current epoch index (for logging).
        is_train       : If True, perform backward pass and scheduler step.
        device         : 'cpu' or 'cuda'.
        log_grad_norms : Log Q/K gradient norms (§2.2 ablation).
        log_confidence : Log prediction confidence of correct token (§2.5).

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    total_confidence = 0.0
    conf_count = 0
    step = 0

    pbar = tqdm(data_iter, desc=f"{'Train' if is_train else 'Val'} Epoch {epoch_num}")

    for src, tgt in pbar:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        # Teacher forcing: feed tgt[:-1] as decoder input, predict tgt[1:]
        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx=PAD_IDX)
        tgt_mask = make_tgt_mask(tgt_in, pad_idx=PAD_IDX)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_in, src_mask, tgt_mask)
            # Flatten for loss: (B * T, V)
            logits_flat = logits.contiguous().view(-1, logits.size(-1))
            tgt_flat    = tgt_out.contiguous().view(-1)

            loss = loss_fn(logits_flat, tgt_flat)

        if is_train:
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), CFG["clip_grad"])

            # Log Q/K gradient norms for §2.2
            if log_grad_norms and step < 1000:
                q_norms, k_norms = [], []
                for layer in model.encoder.layers:
                    if layer.self_attn.W_q.weight.grad is not None:
                        q_norms.append(layer.self_attn.W_q.weight.grad.norm().item())
                    if layer.self_attn.W_k.weight.grad is not None:
                        k_norms.append(layer.self_attn.W_k.weight.grad.norm().item())
                if q_norms:
                    wandb.log({
                        "q_grad_norm": sum(q_norms) / len(q_norms),
                        "k_grad_norm": sum(k_norms) / len(k_norms),
                        "grad_step": step,
                    })

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log({"learning_rate": current_lr})

        # Prediction confidence for §2.5 (softmax prob of correct token)
        if log_confidence:
            with torch.no_grad():
                probs = F.softmax(logits_flat, dim=-1)
                non_pad = tgt_flat != PAD_IDX
                if non_pad.any():
                    correct_probs = probs.gather(1, tgt_flat.unsqueeze(1).clamp(min=0)).squeeze(1)
                    total_confidence += correct_probs[non_pad].sum().item()
                    conf_count += non_pad.sum().item()

        non_pad_count = (tgt_flat != PAD_IDX).sum().item()
        total_loss   += loss.item() * non_pad_count
        total_tokens += non_pad_count
        step         += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total_tokens, 1)

    log_dict = {
        "epoch": epoch_num,
        f"{'train' if is_train else 'val'}_loss": avg_loss,
    }
    if log_confidence and conf_count > 0:
        log_dict["prediction_confidence"] = total_confidence / conf_count
    wandb.log(log_dict)

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model:        Transformer,
    src:          torch.Tensor,
    src_mask:     torch.Tensor,
    max_len:      int,
    start_symbol: int,
    end_symbol:   int,
    device:       str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer (in eval mode).
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    src      = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX).to(device)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model:           Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device:          str = "cpu",
    max_len:         int = 100,
    max_sentences:   int = None,
    beam_size:       int = 5,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader (any batch_size; sentences decoded one by one).
        tgt_vocab       : Vocabulary object supporting lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.
        max_sentences   : If set, stop after this many sentences (for fast val BLEU).

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    import sacrebleu

    model.eval()
    hypotheses: list = []
    references: list = []

    with torch.no_grad():
        _done = False
        for src, tgt in tqdm(test_dataloader, desc="Evaluating BLEU"):
            if _done:
                break
            # greedy_decode expects batch_size=1; iterate sentence by sentence
            for i in range(src.size(0)):
                if max_sentences is not None and len(hypotheses) >= max_sentences:
                    _done = True
                    break
                src_i    = src[i : i + 1].to(device)   # (1, src_len)
                src_mask = make_src_mask(src_i, pad_idx=PAD_IDX).to(device)

                if beam_size > 1:
                    memory = model.encode(src_i, src_mask)
                    ys = model._beam_decode(memory, src_mask,
                                            max_len=max_len, beam_size=beam_size)
                else:
                    ys = greedy_decode(
                        model, src_i, src_mask,
                        max_len=max_len,
                        start_symbol=SOS_IDX,
                        end_symbol=EOS_IDX,
                        device=device,
                    )

                # Decode hypothesis (skip <sos>, stop at <eos>)
                hyp_tokens = []
                for idx in ys[0][1:].tolist():
                    if idx == EOS_IDX:
                        break
                    tok = tgt_vocab.lookup_token(idx)
                    if tok not in ("<pad>",):
                        hyp_tokens.append(tok)
                hypotheses.append(" ".join(hyp_tokens))

                # Decode reference (skip <sos>/<eos>/<pad>)
                ref_tokens = []
                for idx in tgt[i].tolist():
                    if idx in (SOS_IDX, EOS_IDX, PAD_IDX):
                        continue
                    ref_tokens.append(tgt_vocab.lookup_token(idx))
                references.append(" ".join(ref_tokens))

    bleu = sacrebleu.corpus_bleu(hypotheses, [references], force=True)
    return bleu.score


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model:     Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    path:      str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    Saved dict keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config":         model.config,
            "src_vocab": model.src_vocab.to_dict() if model.src_vocab is not None else None,
            "tgt_vocab": model.tgt_vocab.to_dict() if model.tgt_vocab is not None else None,
        },
        path,
    )


def load_checkpoint(
    path:      str,
    model:     Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Returns:
        epoch : The epoch at which the checkpoint was saved.
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    # Restore vocab if stored in checkpoint (enables model.infer() without external setup)
    if model.src_vocab is None and ckpt.get("src_vocab") is not None:
        from dataset import Vocab
        model.src_vocab = Vocab.from_dict(ckpt["src_vocab"])
        model.tgt_vocab = Vocab.from_dict(ckpt["tgt_vocab"])
    return ckpt.get("epoch", 0)


# ══════════════════════════════════════════════════════════════════════
#  ATTENTION MAP VISUALISATION  (§2.3)
# ══════════════════════════════════════════════════════════════════════

def visualize_attention_maps(
    model,
    src_vocab,
    src_spacy,
    device: str,
    sentence_de: str = "Ein Mann in einem blauen Hemd steht vor einem Gebäude .",
) -> None:
    """
    Extract and log per-head attention maps from the last encoder layer (§2.3).
    Logged as W&B Image heatmaps for each attention head.
    """
    model.eval()

    tokens = [t.text.lower() for t in src_spacy.tokenizer(sentence_de)]
    src_idx = [SOS_IDX] + src_vocab.lookup_indices(tokens) + [EOS_IDX]
    src = torch.tensor(src_idx, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)

    with torch.no_grad():
        model.encode(src, src_mask)  # populates attn_weights in last layer

    attn = model.encoder.layers[-1].self_attn.attn_weights  # (1, H, L, L)
    if attn is None:
        return

    attn = attn.cpu().numpy()[0]       # (H, L, L)
    src_labels = ["<sos>"] + tokens + ["<eos>"]
    num_heads = attn.shape[0]

    images = {}
    for h in range(num_heads):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(attn[h], cmap="viridis", vmin=0, vmax=attn[h].max())
        ax.set_xticks(range(len(src_labels)))
        ax.set_yticks(range(len(src_labels)))
        ax.set_xticklabels(src_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(src_labels, fontsize=7)
        ax.set_title(f"Encoder Head {h + 1}")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        images[f"attention_head_{h + 1}"] = wandb.Image(fig)
        plt.close(fig)

    wandb.log({"attention_maps": images})
    print(f"[§2.3] Logged {num_heads} attention head maps to W&B.")


# ══════════════════════════════════════════════════════════════════════
#  CORE TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════

def _run_experiment(
    run_name:       str,
    use_noam:       bool  = True,
    fixed_lr:       float = 1e-4,
    smoothing:      float = 0.1,
    use_scale:      bool  = True,
    use_learned_pe: bool  = False,
    log_grad_norms: bool  = False,
    log_confidence: bool  = False,
    num_epochs:     Optional[int] = None,
    checkpoint_out: str   = "best_checkpoint.pt",
    config_override: dict = None,
) -> Transformer:
    """
    Core experiment runner. Returns the trained model.
    All W&B sections are handled by choosing the right flags.
    """
    from dataset import get_dataloaders

    cfg = {**CFG}
    if config_override:
        cfg.update(config_override)
    if num_epochs is not None:
        cfg["num_epochs"] = num_epochs

    # ── Data FIRST (before CUDA/wandb init to avoid Windows DLL conflict) ──
    print(f"[{run_name}] Loading data...")
    (train_loader, val_loader, test_loader,
     src_vocab, tgt_vocab, src_spacy, tgt_spacy) = get_dataloaders(
        batch_size=cfg["batch_size"],
        max_len=cfg["max_len"],
        min_freq=cfg["min_freq"],
    )
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print(f"  src vocab: {src_vocab_size} | tgt vocab: {tgt_vocab_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{run_name}] Device: {device}")

    # ── W&B init ─────────────────────────────────────────────────────
    wandb.init(
        project=cfg["wandb_project"],
        name=run_name,
        config={
            **cfg,
            "use_noam":        use_noam,
            "fixed_lr":        fixed_lr if not use_noam else None,
            "smoothing":       smoothing,
            "use_scale":       use_scale,
            "use_learned_pe":  use_learned_pe,
            "layer_norm_type": "post",
            "architecture":    "version_A",
            "src_vocab_size":  src_vocab_size,
            "tgt_vocab_size":  tgt_vocab_size,
        },
        reinit="finish_previous",
    )

    # ── Model ────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=cfg["d_model"],
        N=cfg["N"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        use_scale=use_scale,
        use_learned_pe=use_learned_pe,
        checkpoint_path=None,   # training: skip auto-download
    ).to(device)

    # Attach vocab/tokeniser for infer()
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab
    model.src_spacy = src_spacy

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)

    # ── Optimiser ────────────────────────────────────────────────────
    # Adam with paper hyperparameters (β1=0.9, β2=0.98, ε=1e-9)
    # base_lr=1.0 so NoamScheduler directly sets the absolute LR
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0 if use_noam else fixed_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # ── Scheduler ────────────────────────────────────────────────────
    if use_noam:
        scheduler = NoamScheduler(
            optimizer,
            d_model=cfg["d_model"],
            warmup_steps=cfg["warmup_steps"],
        )
    else:
        scheduler = None  # constant LR

    # ── Loss function ────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=PAD_IDX,
        smoothing=smoothing,
    )

    # ── Training loop ────────────────────────────────────────────────
    best_val_bleu = 0.0
    best_ckpt     = checkpoint_out

    for epoch in range(cfg["num_epochs"]):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
            log_grad_norms=log_grad_norms,
            log_confidence=log_confidence,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
            log_confidence=log_confidence,
        )

        # Compute validation BLEU every epoch (capped at 256 sentences for speed)
        val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device,
                                 max_len=50, max_sentences=256, beam_size=1)
        wandb.log({"val_bleu": val_bleu, "epoch": epoch})

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_bleu={val_bleu:.2f}"
        )

        # Save best model
        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            if scheduler is not None:
                save_checkpoint(model, optimizer, scheduler, epoch, best_ckpt)
            else:
                _dummy_sched = type("_S", (), {"state_dict": lambda s: {}})()
                save_checkpoint(model, optimizer, _dummy_sched, epoch, best_ckpt)
            print(f"  → Checkpoint saved (val_bleu={val_bleu:.2f})")

    # ── Test BLEU on best checkpoint ─────────────────────────────────
    load_checkpoint(best_ckpt, model)
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device, max_len=100)
    wandb.log({"test_bleu": test_bleu})
    print(f"Test BLEU: {test_bleu:.2f}")

    wandb.finish()
    return model, src_vocab, tgt_vocab, src_spacy


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC EXPERIMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Main experiment — called by autograder and covers §2.1 (Noam),
    §2.3 (attention maps), and §2.5 (label smoothing baseline).
    """
    model, src_vocab, _, src_spacy = _run_experiment(
        run_name="A_main_noam_ls01",
        use_noam=True,
        smoothing=0.1,
        use_scale=True,
        use_learned_pe=False,
        log_confidence=True,
        checkpoint_out="best_checkpoint_A_main.pt",
    )

    # §2.3: Attention map visualisation (re-open a W&B run for logging)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from dataset import get_dataloaders
    wandb.init(
        project=CFG["wandb_project"],
        name="A_attention_maps",
        reinit="finish_previous",
    )
    visualize_attention_maps(model, src_vocab, src_spacy, device)
    wandb.finish()


def run_fixed_lr_experiment() -> None:
    """§2.1: Fixed learning rate baseline (no Noam scheduler)."""
    _run_experiment(
        run_name="A_fixed_lr_1e4",
        use_noam=False,
        fixed_lr=1e-4,
        smoothing=0.1,
        use_scale=True,
        use_learned_pe=False,
        checkpoint_out="best_checkpoint_A_fixedlr.pt",
    )


def run_no_scale_experiment() -> None:
    """§2.2: Attention without 1/√d_k scaling — logs Q/K gradient norms."""
    _run_experiment(
        run_name="A_no_attention_scale",
        use_noam=True,
        smoothing=0.1,
        use_scale=False,          # disable sqrt(d_k) scaling
        use_learned_pe=False,
        log_grad_norms=True,      # log Q/K grad norms first 1000 steps
        num_epochs=5,             # shorter run for comparison
        checkpoint_out="best_checkpoint_A_noscale.pt",
    )
    # Also run with scale for the same number of epochs for fair comparison
    _run_experiment(
        run_name="A_with_attention_scale",
        use_noam=True,
        smoothing=0.1,
        use_scale=True,
        use_learned_pe=False,
        log_grad_norms=True,
        num_epochs=5,
        checkpoint_out="best_checkpoint_A_scale.pt",
    )


def run_learned_pe_experiment() -> None:
    """§2.4: Learned positional encoding vs sinusoidal."""
    _run_experiment(
        run_name="A_learned_pe",
        use_noam=True,
        smoothing=0.1,
        use_scale=True,
        use_learned_pe=True,      # learned PE
        checkpoint_out="best_checkpoint_A_learnedpe.pt",
    )
    # Sinusoidal baseline already covered by main run;
    # run a shorter one here for a clean side-by-side comparison
    _run_experiment(
        run_name="A_sinusoidal_pe",
        use_noam=True,
        smoothing=0.1,
        use_scale=True,
        use_learned_pe=False,
        num_epochs=10,
        checkpoint_out="best_checkpoint_A_sinpe.pt",
    )


def run_no_smoothing_experiment() -> None:
    """§2.5: Standard cross-entropy (smoothing=0) vs label smoothing."""
    _run_experiment(
        run_name="A_no_label_smoothing",
        use_noam=True,
        smoothing=0.0,            # standard cross-entropy
        use_scale=True,
        use_learned_pe=False,
        log_confidence=True,      # track prediction confidence
        checkpoint_out="best_checkpoint_A_nols.pt",
    )


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DA6401 Assignment 3 — Version A")
    parser.add_argument(
        "--experiment",
        type=str,
        default="main",
        choices=["main", "fixed_lr", "no_scale", "learned_pe", "no_smoothing", "all"],
        help="Which W&B experiment to run (default: main)",
    )
    args = parser.parse_args()

    if args.experiment == "main":
        run_training_experiment()
    elif args.experiment == "fixed_lr":
        run_fixed_lr_experiment()
    elif args.experiment == "no_scale":
        run_no_scale_experiment()
    elif args.experiment == "learned_pe":
        run_learned_pe_experiment()
    elif args.experiment == "no_smoothing":
        run_no_smoothing_experiment()
    elif args.experiment == "all":
        print("=== Running ALL experiments (§2.1–§2.5) ===")
        run_training_experiment()
        run_fixed_lr_experiment()
        run_no_scale_experiment()
        run_learned_pe_experiment()
        run_no_smoothing_experiment()

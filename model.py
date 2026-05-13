"""
model.py — Transformer Architecture  (Version A: Post-LayerNorm)
DA6401 Assignment 3: "Attention Is All You Need"

Architecture choice: Post-LayerNorm  x = LayerNorm(x + sublayer(x))
as in the original Vaswani et al. (2017) paper.

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Special token indices (must match dataset.py) ──────────────────────
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    # Scaled dot-product scores: (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = F.softmax(scores, dim=-1)
    # Replace NaN rows (all-masked) with 0 so arithmetic stays valid
    attn_w = torch.nan_to_num(attn_w, nan=0.0)

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → PAD position (masked out)
        False → real token
    """
    # (batch, src_len) → (batch, 1, 1, src_len)
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build combined padding + causal (look-ahead) mask for the decoder.

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → masked out (PAD or future token)
    """
    batch_size, tgt_len = tgt.shape

    # Padding mask: True for PAD tokens
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,tgt_len)

    # Causal mask: upper triangle (future tokens)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)  # (1,1,tgt_len,tgt_len)

    return pad_mask | causal_mask  # broadcasts to (B,1,tgt_len,tgt_len)


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
        use_scale (bool) : Whether to divide by sqrt(d_k). Default True.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_scale: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.use_scale = use_scale

        # Projection matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        # Stored after each forward — used for visualisation (§2.3)
        self.attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor, batch: int) -> torch.Tensor:
        """(B, seq, d_model) → (B, heads, seq, d_k)"""
        return x.view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        B = query.size(0)

        Q = self._split_heads(self.W_q(query), B)  # (B, H, seq_q, d_k)
        K = self._split_heads(self.W_k(key),   B)  # (B, H, seq_k, d_k)
        V = self._split_heads(self.W_v(value), B)  # (B, H, seq_k, d_k)

        if self.use_scale:
            out, w = scaled_dot_product_attention(Q, K, V, mask)
        else:
            # Raw dot-product without 1/√d_k  (ablation experiment §2.2)
            scores = torch.matmul(Q, K.transpose(-2, -1))
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            w = F.softmax(scores, dim=-1)
            w = torch.nan_to_num(w, nan=0.0)
            out = torch.matmul(w, V)

        # Store attention weights for visualisation (detach to free graph)
        self.attn_weights = w.detach()

        # Merge heads: (B, H, seq_q, d_k) → (B, seq_q, d_model)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Registered as a buffer (not a trainable parameter).

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build PE table of shape (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # div_term[i] = 10000^(2i / d_model) inverted using log-space
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)                           # (1, max_len, d_model)

        # Buffer: saved with model state but not a learnable parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]
        Returns:
            Tensor of same shape with positional encodings added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  LEARNED POSITIONAL ENCODING  (experiment §2.4)
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding using nn.Embedding (§2.4 ablation).
    Replaces sinusoidal PE as a drop-in; positions are learnable parameters.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.embedding(positions)
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:
        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  (Post-LayerNorm)
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer (Post-LN):
        x → Self-Attention → Add & Norm → FFN → Add & Norm

    Post-LN: LayerNorm is applied AFTER the residual addition, matching
    the original Vaswani et al. (2017) paper exactly.
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        d_ff:      int,
        dropout:   float = 0.1,
        use_scale: bool  = True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Post-LN: norm( x + sublayer(x) )
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER  (Post-LayerNorm)
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer (Post-LN):
        x → Masked Self-Attn → Add & Norm
          → Cross-Attn(memory) → Add & Norm
          → FFN → Add & Norm
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        d_ff:      int,
        dropout:   float = 0.1,
        use_scale: bool  = True,
    ) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Masked self-attention (Post-LN)
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        # Cross-attention on encoder memory (Post-LN)
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        # Feed-forward (Post-LN)
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size  : Source vocabulary size.
        tgt_vocab_size  : Target vocabulary size.
        d_model         : Model dimensionality (default 512).
        N               : Number of encoder/decoder layers (default 6).
        num_heads       : Number of attention heads (default 8).
        d_ff            : FFN inner dimensionality (default 2048).
        dropout         : Dropout probability (default 0.1).
        checkpoint_path : If provided, downloads checkpoint from GDrive
                          and loads weights (used by autograder).
        use_scale       : Use 1/√d_k scaling in attention (default True).
        use_learned_pe  : Use learned positional encoding (default False).
    """

    # ── UPDATE this ID after uploading your trained checkpoint to Google Drive ──
    GDRIVE_FILE_ID: str = "1ZICakbwgqSBNftlYRyv9EiN4N9hslzrf"
    _DEFAULT_CKPT: str = "best_checkpoint_A_main.pt"

    def __init__(
        self,
        src_vocab_size: int   = 7853,
        tgt_vocab_size: int   = 5893,
        d_model:        int   = 512,
        N:              int   = 6,
        num_heads:      int   = 8,
        d_ff:           int   = 2048,
        dropout:        float = 0.1,
        checkpoint_path: Optional[str] = "auto",
        use_scale:      bool  = True,
        use_learned_pe: bool  = False,
    ) -> None:
        super().__init__()

        # Store config for save_checkpoint
        self.config = dict(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model, N=N, num_heads=num_heads,
            d_ff=d_ff, dropout=dropout,
        )
        self.d_model = d_model

        # ── Embeddings ────────────────────────────────────────────────
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)

        # ── Positional encoding ───────────────────────────────────────
        PE = LearnedPositionalEncoding if use_learned_pe else PositionalEncoding
        self.src_pe = PE(d_model, dropout)
        self.tgt_pe = PE(d_model, dropout)

        # ── Encoder / Decoder stacks ─────────────────────────────────
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout, use_scale)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout, use_scale)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        # ── Output projection ─────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # ── Xavier weight initialisation ──────────────────────────────
        self._init_weights()
        # Weight tying: share embedding and output projection (Vaswani et al. SS3.4)
        self.output_proj.weight = self.tgt_embed.weight

        # ── Vocab / tokeniser / checkpoint (all loaded eagerly per spec) ──
        self.src_vocab = None
        self.tgt_vocab = None
        self.src_spacy = None

        # Resolve checkpoint path: "auto" = download from GDrive to default name
        _ckpt_path = checkpoint_path
        if _ckpt_path == "auto":
            if self.GDRIVE_FILE_ID != "YOUR_GDRIVE_FILE_ID_HERE":
                _ckpt_path = self._DEFAULT_CKPT
                if not os.path.exists(_ckpt_path):
                    gdown.download(id=self.GDRIVE_FILE_ID, output=_ckpt_path, quiet=False)
            else:
                _ckpt_path = None  # no real ID yet (pre-upload training)

        if _ckpt_path is not None and os.path.exists(_ckpt_path):
            _ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
            self.load_state_dict(_ckpt["model_state_dict"])
            # Restore vocab so infer() works without re-building the dataset
            if _ckpt.get("src_vocab") is not None:
                from dataset import Vocab
                self.src_vocab = Vocab.from_dict(_ckpt["src_vocab"])
                self.tgt_vocab = Vocab.from_dict(_ckpt["tgt_vocab"])
            # Load German spaCy tokeniser (with download fallback)
            self.src_spacy = self._load_spacy_de()

    @staticmethod
    def _load_spacy_de():
        """Load de_core_news_sm; download it silently if not installed."""
        import spacy, sys
        try:
            return spacy.load("de_core_news_sm")
        except OSError:
            try:
                import subprocess
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "de_core_news_sm"],
                    check=True, capture_output=True,
                )
                return spacy.load("de_core_news_sm")
            except Exception:
                return spacy.blank("de")

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ────────────────────────────────────────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.src_pe(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.tgt_pe(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.output_proj(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def _ensure_vocab(self) -> None:
        """Auto-load vocab + spaCy from the dataset if not already set."""
        if self.src_vocab is None:
            import os
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            import datasets as _ds  # noqa: F401 — must load before torch CUDA
            from dataset import get_dataloaders
            _, _, _, src_vocab, tgt_vocab, src_spacy, _ = get_dataloaders(
                batch_size=64, num_workers=0
            )
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            if self.src_spacy is None:
                self.src_spacy = src_spacy
        if self.src_spacy is None:
            self.src_spacy = self._load_spacy_de()

    def _beam_decode(
        self,
        memory:    torch.Tensor,
        src_mask:  torch.Tensor,
        max_len:   int = 100,
        beam_size: int = 5,
        alpha:     float = 0.6,
    ) -> torch.Tensor:
        """Length-normalised beam search. Returns best token-id sequence (1, T)."""
        device = memory.device

        init_ys = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
        # (ys_tensor, cumulative_log_prob)
        beams: list = [(init_ys, 0.0)]
        finished: list = []

        with torch.no_grad():
            for _ in range(max_len):
                if not beams:
                    break
                candidates: list = []
                for ys, score in beams:
                    tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX)
                    logits   = self.decode(memory, src_mask, ys, tgt_mask)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)[0]
                    topk_lp, topk_ids = log_probs.topk(beam_size)
                    for lp, tok in zip(topk_lp.tolist(), topk_ids.tolist()):
                        new_ys    = torch.cat([ys, torch.tensor([[tok]], device=device)], dim=1)
                        new_score = score + lp
                        if tok == EOS_IDX:
                            finished.append((new_ys, new_score))
                        else:
                            candidates.append((new_ys, new_score))

                def _norm(item: tuple) -> float:
                    ys, sc = item
                    return sc / max(1, (ys.size(1) - 1)) ** alpha

                candidates.sort(key=_norm, reverse=True)
                beams = candidates[:beam_size]

        finished.extend(beams)
        if not finished:
            return init_ys
        best_ys, _ = max(finished, key=lambda x: x[1] / max(1, (x[0].size(1) - 1)) ** alpha)
        return best_ys

    def infer(self, src_sentence: str) -> str:
        """
        Translate a German sentence to English using beam search (beam=5).

        Vocab and spaCy tokeniser are loaded automatically if not already set
        (required by the autograder which creates Transformer() with no args).

        Args:
            src_sentence: Raw German text.

        Returns:
            Translated English string.
        """
        self._ensure_vocab()
        self.eval()
        device = next(self.parameters()).device

        # Tokenise & numericalize (same pipeline as training data)
        tokens  = [t.text.lower() for t in self.src_spacy.tokenizer(src_sentence)]
        src_idx = [SOS_IDX] + self.src_vocab.lookup_indices(tokens) + [EOS_IDX]
        src     = torch.tensor(src_idx, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = make_src_mask(src, pad_idx=PAD_IDX)

        with torch.no_grad():
            memory = self.encode(src, src_mask)

        ys = self._beam_decode(memory, src_mask, max_len=100, beam_size=5)

        out_tokens = [
            self.tgt_vocab.lookup_token(idx.item())
            for idx in ys[0][1:]   # skip <sos>
        ]
        out_tokens = [t for t in out_tokens if t not in ("<eos>", "<pad>")]
        return " ".join(out_tokens)

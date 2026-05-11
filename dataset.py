"""
dataset.py — Multi30k Dataset Loading and Vocabulary Construction
DA6401 Assignment 3: "Attention Is All You Need"

Loads the bentrevett/multi30k dataset from HuggingFace, tokenises with spaCy,
builds shared source/target vocabularies, and exposes PyTorch-compatible
Dataset and DataLoader helpers.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import load_dataset as _hf_load_dataset  # must be before torch

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from typing import List, Tuple, Optional
from tqdm import tqdm
import spacy

# ── Special token indices (kept consistent with model.py) ─────────────
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]


# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY CLASS
# ══════════════════════════════════════════════════════════════════════

class Vocab:
    """Simple string↔index vocabulary with special tokens."""

    def __init__(self, stoi: dict, itos: dict) -> None:
        self.stoi = stoi          # str  → int
        self.itos = itos          # int  → str

    def __len__(self) -> int:
        return len(self.stoi)

    # Autograder-compatible lookup helpers
    def lookup_token(self, idx: int) -> str:
        return self.itos.get(int(idx), "<unk>")

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, UNK_IDX) for t in tokens]

    def to_dict(self) -> dict:
        return {"stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        itos = {int(k): v for k, v in d["itos"].items()}
        return cls(d["stoi"], itos)


# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY BUILDER
# ══════════════════════════════════════════════════════════════════════

def _build_vocab(token_lists: List[List[str]], min_freq: int = 2) -> Vocab:
    """
    Build a Vocab from a list of token lists.
    Vocabulary includes SPECIALS at fixed indices 0-3.
    """
    counter: Counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    stoi = {tok: idx for idx, tok in enumerate(SPECIALS)}
    for tok, freq in counter.most_common():
        if freq >= min_freq and tok not in stoi:
            stoi[tok] = len(stoi)

    itos = {idx: tok for tok, idx in stoi.items()}
    return Vocab(stoi, itos)


# ══════════════════════════════════════════════════════════════════════
#  MULTI30K DATASET
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    Multi30k German→English dataset (bentrevett/multi30k on HuggingFace).

    Provides PyTorch Dataset interface; vocabularies are built on the
    training split and shared with validation/test.

    Args:
        split      : 'train', 'validation', or 'test'.
        src_vocab  : Pre-built Vocab for German (None = build from data).
        tgt_vocab  : Pre-built Vocab for English (None = build from data).
        src_spacy  : Loaded spaCy German model (None = load internally).
        tgt_spacy  : Loaded spaCy English model (None = load internally).
        max_len    : Truncate sequences to this length (tokens + specials).
        min_freq   : Minimum token frequency to include in vocab.
    """

    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[Vocab] = None,
        tgt_vocab: Optional[Vocab] = None,
        src_spacy=None,
        tgt_spacy=None,
        max_len: int = 128,
        min_freq: int = 2,
    ) -> None:
        self.split = split
        self.max_len = max_len

        # ── Load spaCy tokenisers ─────────────────────────────────────
        self.src_spacy = src_spacy or _load_spacy("de_core_news_sm")
        self.tgt_spacy = tgt_spacy or _load_spacy("en_core_web_sm")

        # ── Load raw HuggingFace dataset ─────────────────────────────
        raw = _hf_load_dataset("bentrevett/multi30k", split=split)

        # ── Tokenise ─────────────────────────────────────────────────
        self._src_tokens: List[List[str]] = [
            self._tok_de(ex["de"])
            for ex in tqdm(raw, desc=f"  Tokenising DE [{split}]", leave=False)
        ]
        self._tgt_tokens: List[List[str]] = [
            self._tok_en(ex["en"])
            for ex in tqdm(raw, desc=f"  Tokenising EN [{split}]", leave=False)
        ]

        # ── Build or reuse vocabulary ─────────────────────────────────
        if src_vocab is None:
            self.src_vocab = _build_vocab(self._src_tokens, min_freq)
        else:
            self.src_vocab = src_vocab

        if tgt_vocab is None:
            self.tgt_vocab = _build_vocab(self._tgt_tokens, min_freq)
        else:
            self.tgt_vocab = tgt_vocab

        # ── Numericalize ─────────────────────────────────────────────
        self.src_data: List[List[int]] = [
            ([SOS_IDX] + self.src_vocab.lookup_indices(toks) + [EOS_IDX])[: max_len]
            for toks in self._src_tokens
        ]
        self.tgt_data: List[List[int]] = [
            ([SOS_IDX] + self.tgt_vocab.lookup_indices(toks) + [EOS_IDX])[: max_len]
            for toks in self._tgt_tokens
        ]

    # ── Template-required public API ──────────────────────────────────

    def build_vocab(self):
        """Returns (src_vocab, tgt_vocab) — already built in __init__."""
        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        """Returns (src_data, tgt_data) as lists of index lists."""
        return self.src_data, self.tgt_data

    # ── PyTorch Dataset ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = torch.tensor(self.src_data[idx], dtype=torch.long)
        tgt = torch.tensor(self.tgt_data[idx], dtype=torch.long)
        return src, tgt

    # ── Private helpers ───────────────────────────────────────────────

    def _tok_de(self, text: str) -> List[str]:
        return [t.text.lower() for t in self.src_spacy.tokenizer(text)]

    def _tok_en(self, text: str) -> List[str]:
        return [t.text.lower() for t in self.tgt_spacy.tokenizer(text)]


# ══════════════════════════════════════════════════════════════════════
#  COLLATE & DATALOADERS
# ══════════════════════════════════════════════════════════════════════

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of (src, tgt) pairs to equal length."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def get_dataloaders(
    batch_size: int = 64,
    num_workers: int = 0,
    max_len: int = 128,
    min_freq: int = 2,
):
    """
    Build and return train / validation / test DataLoaders for Multi30k.

    Vocabulary is built from the training split only and shared with
    validation and test splits (no data leakage).

    Returns:
        train_loader, val_loader, test_loader,
        src_vocab, tgt_vocab, src_spacy, tgt_spacy
    """
    # Load spaCy once and share across splits (avoids repeated model loads)
    print("  Loading spaCy models...", flush=True)
    src_spacy = _load_spacy("de_core_news_sm")
    tgt_spacy = _load_spacy("en_core_web_sm")

    # Build vocab from training data only
    print("  Building train dataset + vocab (tokenising 29k sentences)...", flush=True)
    train_ds = Multi30kDataset(
        split="train",
        src_spacy=src_spacy,
        tgt_spacy=tgt_spacy,
        max_len=max_len,
        min_freq=min_freq,
    )
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    print("  Building validation dataset...", flush=True)
    val_ds = Multi30kDataset(
        split="validation",
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_spacy=src_spacy,
        tgt_spacy=tgt_spacy,
        max_len=max_len,
    )
    print("  Building test dataset...", flush=True)
    test_ds = Multi30kDataset(
        split="test",
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_spacy=src_spacy,
        tgt_spacy=tgt_spacy,
        max_len=max_len,
    )

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin,
    )
    # test: batch_size=1 so greedy_decode can process sentence-by-sentence
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(
        f"  Done. src_vocab={len(src_vocab)} tgt_vocab={len(tgt_vocab)} "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
        flush=True,
    )
    return (
        train_loader, val_loader, test_loader,
        src_vocab, tgt_vocab, src_spacy, tgt_spacy,
    )


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL HELPER
# ══════════════════════════════════════════════════════════════════════

def _load_spacy(model_name: str):
    """Load a spaCy model, giving a clear error if not downloaded."""
    try:
        return spacy.load(model_name)
    except OSError:
        raise OSError(
            f"spaCy model '{model_name}' not found. "
            f"Run:  python -m spacy download {model_name}"
        )

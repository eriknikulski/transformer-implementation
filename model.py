import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    """One single self-attention head"""

    def __init__(self, embd_size: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(embd_size, head_size, bias=False)
        self.key = nn.Linear(embd_size, head_size, bias=False)
        self.value = nn.Linear(embd_size, head_size, bias=False)

        self.register_buffer(
            'mask',
            torch.tril(torch.ones((block_size, block_size), dtype=torch.bool)).logical_not()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, hs = x.shape

        q = self.query(x)               # (B, T, hs)
        k = self.key(x)                 # (B, T, hs)

        wei = q @ k.transpose(-2, -1) * 1 / math.sqrt(hs)               # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.mask[:T, :T], float('-inf'))         # (B, T, T)
        wei = F.softmax(wei, dim=-1)    # (B, T, T)
        wei = self.dropout(wei)         # (B, T, T)

        v = self.value(x)               # (B, T, hs)
        out = wei @ v                   # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out


class MaskedMultiHeadAttention(nn.Module):
    """Masked multi-head self-attention"""

    def __init__(self, embd_size: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()

        head_size = embd_size // n_heads

        self.heads = nn.ModuleList(
            [AttentionHead(embd_size, head_size, block_size, dropout) for _ in range(n_heads)]
        )

        self.proj = nn.Linear(head_size * n_heads, embd_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)       # (B, T, hs * n_heads)
        out = self.proj(out)        # (B, T, C)
        out = self.dropout(out)     # (B, T, C)
        return out


class FeedForward(nn.Module):
    """Two layer feed-forward neural network"""

    def __init__(self, embd_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_size, 4 * embd_size),
            nn.ReLU(),
            nn.Linear(4 * embd_size, embd_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """One single transformer block"""
    def __init__(self, embd_size: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()

        self.mmh_attn = MaskedMultiHeadAttention(embd_size, n_heads, block_size, dropout)
        self.ffw = FeedForward(embd_size, dropout)

        self.ln1 = nn.LayerNorm(embd_size)
        self.ln2 = nn.LayerNorm(embd_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mmh_attn(self.ln1(x))      # (B, T, C)
        x = x + self.ffw(self.ln2(x))           # (B, T, C)
        return x


class Transformer(nn.Module):
    """Transformer"""
    def __init__(self, n_blocks: int, vocab_size: int, embd_size: int, block_size: int, n_heads: int, dropout: float):
        super().__init__()
        self.block_size = block_size

        self.tok_embd_table = nn.Embedding(vocab_size, embd_size)
        self.pos_embd_table = nn.Embedding(block_size, embd_size)

        self.blocks = nn.Sequential(
            *[TransformerBlock(embd_size, n_heads, block_size, dropout) for _ in range(n_blocks)]
        )

        self.ln = nn.LayerNorm(embd_size)

        self.lm_head = nn.Linear(embd_size, vocab_size)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        B, T = x.shape

        tok_embd = self.tok_embd_table(x)                       # (B, T, C)
        pos_embd = self.pos_embd_table(torch.arange(T))         # (T, C)
        out = tok_embd + pos_embd                               # (B, T, C)

        out = self.blocks(out)                  # (B, T, C)
        out = self.ln(out)                      # (B, T, C)

        logits = self.lm_head(out)              # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return out, loss

    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)           # (B, T, C)
            logits = logits[:, -1, :]               # (B, C)
            probs = F.softmax(logits, dim=-1)       # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx

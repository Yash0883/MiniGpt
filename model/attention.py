import math

import torch
import torch.nn as nn

import config


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention with batched heads."""

    def __init__(self, embed_dim):
        super().__init__()

        assert embed_dim % config.NUM_HEADS == 0, \
            "embed_dim must be divisible by NUM_HEADS"

        self.num_heads = config.NUM_HEADS
        self.head_dim = embed_dim // self.num_heads

        # One projection computes Q, K, and V for every head at once.
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        self.output_projection = nn.Linear(
            embed_dim,
            embed_dim
        )

        # Reuse the mask on every forward pass instead of allocating it per
        # head and per batch.
        causal_mask = torch.tril(
            torch.ones(config.MAX_SEQ_LEN, config.MAX_SEQ_LEN, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        if seq_len > self.causal_mask.size(0):
            raise ValueError(
                f"Sequence length {seq_len} exceeds MAX_SEQ_LEN "
                f"({self.causal_mask.size(0)})"
            )

        # (B, T, 3C) -> (3, B, heads, T, head_dim)
        qkv = self.qkv(x).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(0)

        # (B, heads, T, D) @ (B, heads, D, T)
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output = weights @ value

        # (B, heads, T, D) -> (B, T, embed_dim)
        output = output.transpose(1, 2).contiguous().reshape(
            batch_size,
            seq_len,
            embed_dim,
        )

        return self.output_projection(output)

"""
gpt.py

MiniGPT Model

Architecture:

Input Tokens
      │
      ▼
Token Embedding
      │
      ▼
Transformer Blocks
      │
      │  (RoPE is applied to Q and K inside attention)
      ▼
      │
      ▼
      RMSNorm
      │
      ▼
LM Head (Weight Tied)
      │
      ▼
Vocabulary Logits
"""

import torch.nn as nn
from config import GPTConfig

from .embeddings import TokenEmbedding
from .transformer_block import TransformerBlock


class MiniGPT(nn.Module):

    def __init__(self, config=None):
        super().__init__()

        self.config = config or GPTConfig()

        # ---------------------------------
        # Token Embedding
        # ---------------------------------
        self.token_embedding = TokenEmbedding(
            self.config.vocab_size,
            self.config.embed_dim
        )

        # ---------------------------------
        # Transformer Blocks
        # ---------------------------------
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(self.config)
                for _ in range(self.config.num_layers)
            ]
        )

        # ---------------------------------
        # Final RMSNorm
        # ---------------------------------
        self.norm = nn.RMSNorm(self.config.embed_dim, eps=1e-5)

        # ---------------------------------
        # Language Modeling Head
        # ---------------------------------
        self.lm_head = nn.Linear(
            self.config.embed_dim,
            self.config.vocab_size,
            bias=False
        )

        # ---------------------------------
        # Weight Tying
        # GPT-2 Style
        # ---------------------------------
        self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, token_ids, past_key_values=None, use_cache=False):

        # Token Embeddings
        token = self.token_embedding(token_ids)

        # Transformer. The regular path keeps the original simple API used
        # during training. The cache path is used by autoregressive generation.
        if not use_cache:
            x = self.blocks(token)
        else:
            if past_key_values is None:
                past_key_values = [None] * len(self.blocks)

            if len(past_key_values) != len(self.blocks):
                raise ValueError("One KV cache entry is required per transformer block")

            position_offset = 0
            if past_key_values[0] is not None:
                position_offset = past_key_values[0][0].size(-2)

            presents = []
            x = token
            for block, past_key_value in zip(self.blocks, past_key_values):
                x, present_key_value = block(
                    x,
                    past_key_value=past_key_value,
                    position_offset=position_offset,
                    use_cache=True,
                )
                presents.append(present_key_value)

        # Final Normalization
        x = self.norm(x)

        # Vocabulary Logits
        logits = self.lm_head(x)

        if use_cache:
            return logits, presents

        return logits

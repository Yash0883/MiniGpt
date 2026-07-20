import torch.nn as nn
import config

from .embeddings import TokenEmbedding, PositionalEmbedding
from .transformer_block import TransformerBlock

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            config.VOCAB_SIZE,
            config.EMBED_DIM
        )

        self.position_embedding = PositionalEmbedding(
            config.MAX_SEQ_LEN,
            config.EMBED_DIM
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(config.EMBED_DIM)
                for _ in range(config.NUM_LAYERS)
            ]

        )

        self.ln = nn.LayerNorm(config.EMBED_DIM)

        self.lm_head = nn.Linear(
            config.EMBED_DIM,
            config.VOCAB_SIZE
        )

    def forward(self, token_ids):

        token = self.token_embedding(token_ids)
        position = self.position_embedding(token_ids)

        x = token+position
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        return logits

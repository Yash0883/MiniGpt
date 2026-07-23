import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward
from config import GPTConfig

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        embed_dim = config.embed_dim
        self.norm1 = nn.RMSNorm(embed_dim, eps=1e-5)
        self.attention = MultiHeadAttention(config)
        self.norm2 = nn.RMSNorm(embed_dim, eps=1e-5)
        self.feedforward = FeedForward(config)

    def forward(
        self,
        x,
        past_key_value=None,
        position_offset=0,
        use_cache=False,
    ):

        attention_output = self.attention(
            self.norm1(x),
            past_key_value=past_key_value,
            position_offset=position_offset,
            use_cache=use_cache,
        )

        if use_cache:
            attention_output, present_key_value = attention_output

        x = x + attention_output
        x = x + self.feedforward(self.norm2(x))

        if use_cache:
            return x, present_key_value

        return x

import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.feedforward = FeedForward(embed_dim)

    def forward(self, x):

        x = x+self.attention(self.ln1(x))
        x = x+self.feedforward(self.ln2(x))

        return x
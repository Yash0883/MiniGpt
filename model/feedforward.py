import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig


class FeedForward(nn.Module):
    """SwiGLU feed-forward network used inside a transformer block."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        embed_dim = config.embed_dim
        hidden_dim = config.ffn_hidden

        # SwiGLU has two input projections: one produces the gate and the
        # other produces the values that the gate controls.
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)

        # Element-wise gating, followed by projection back to embed_dim.
        return self.down_proj(gate * value)

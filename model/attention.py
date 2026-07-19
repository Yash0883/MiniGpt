import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(Q.size(-1))
        weights = torch.softmax(scores, dim=-1)
        output = weights @ V

        return output




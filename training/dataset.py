"""
dataset.py

Lazy Dataset for next-token prediction.

Instead of creating every training sample in advance,
we store only the token IDs and slice them on demand.
"""

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, token_ids, context_length):

        self.token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):

        return self.token_ids.numel() - self.context_length

    def __getitem__(self, idx):

        x = self.token_ids[
            idx : idx + self.context_length
        ]

        y = self.token_ids[
            idx + 1 : idx + self.context_length + 1
        ]

        return x, y

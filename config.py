import torch


class GPTConfig:
    """Configuration for one MiniGPT model and training run."""

    def __init__(
        self,
        vocab_size=8000,
        max_seq_len=128,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        batch_size=32,
        learning_rate=3e-4,
        epochs=10,
        device=None,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_hidden = embed_dim * 4
        self.dropout = 0.1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device or self._default_device()

    @staticmethod
    def _default_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

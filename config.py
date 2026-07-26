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
        # The existing Shakespeare checkpoint completed epoch 10. Running to
        # epoch 20 performs ten additional fine-tuning epochs on TinyStories.
        epochs=20,
        warmup_steps=500,
        min_learning_rate=1e-6,
        grad_clip=1.0,
        validation_split=0.1,
        log_interval=100,
        checkpoint_dir="checkpoints",
        # Epochs 1-19 completed; resume with the final epoch 20.
        resume_from="checkpoints/minigpt_epoch_19.pth",
        dataset_name="roneneldan/TinyStories",
        dataset_config=None,
        dataset_split="train",
        text_column="text",
        max_train_tokens=1_000_000,
        max_documents=None,
        tokenizer_max_documents=100_000,
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
        self.warmup_steps = warmup_steps
        self.min_learning_rate = min_learning_rate
        self.grad_clip = grad_clip
        self.validation_split = validation_split
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir
        self.resume_from = resume_from
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.text_column = text_column
        self.max_train_tokens = max_train_tokens
        self.max_documents = max_documents
        self.tokenizer_max_documents = tokenizer_max_documents
        self.device = device or self._default_device()

    def to_dict(self):
        """Return checkpoint-safe configuration values."""
        return {
            key: str(value) if key == "device" else value
            for key, value in self.__dict__.items()
        }

    @staticmethod
    def _default_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import GPTConfig


cfg = GPTConfig()

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=cfg.vocab_size,
    special_tokens = [
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]"
    ]
)

load_kwargs = {"split": cfg.dataset_split}
if cfg.dataset_config is not None:
    load_kwargs["name"] = cfg.dataset_config

dataset = load_dataset(cfg.dataset_name, **load_kwargs)


def text_iterator():
    for index, row in enumerate(dataset):
        if (
            cfg.tokenizer_max_documents is not None
            and index >= cfg.tokenizer_max_documents
        ):
            break

        text = row.get(cfg.text_column)
        if text:
            yield text


# This overwrites tokenizer/tokenizer.json with a tokenizer for this dataset.
tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

tokenizer.save("tokenizer/tokenizer.json")

print(f"Tokenizer trained successfully: {tokenizer.get_vocab_size()} tokens")

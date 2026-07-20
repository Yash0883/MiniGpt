from tokenizers import Tokenizer 
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size = 8000,
    special_tokens = [
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]"
    ]
)

tokenizer.train(
    ["/Users/yashh/Desktop/MiniGpt/data/raw/shakespeare.txt"],
    trainer
)

tokenizer.save("tokenizer/tokenizer.json")

print("Tokenizer Trained Successfull")
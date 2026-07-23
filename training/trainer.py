"""
trainer.py

Purpose:
Train MiniGPT using next-token prediction.

Pipeline:
Raw Text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Dataset
    ↓
DataLoader
    ↓
MiniGPT
    ↓
CrossEntropy Loss
    ↓
Backpropagation
    ↓
Weight Update
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import time 

from config import GPTConfig

from model.gpt import MiniGPT
from training.dataset import TextDataset


cfg = GPTConfig()


# --------------------------------------------------
# Load Tokenizer
# --------------------------------------------------

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")


# --------------------------------------------------
# Load Training Text
# --------------------------------------------------

# Change this path if you use another dataset
with open(
    "/Users/yashh/Desktop/MiniGpt/data/raw/shakespeare.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

# --------------------------------------------------
# Convert Text → Token IDs
# --------------------------------------------------

token_ids = tokenizer.encode(text).ids

print(f"Total Tokens : {len(token_ids)}")


# --------------------------------------------------
# Create Dataset
# --------------------------------------------------

dataset = TextDataset(
    token_ids=token_ids,
    context_length=cfg.max_seq_len
)


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

loader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True
)


# --------------------------------------------------
# Create Model
# --------------------------------------------------

model = MiniGPT(cfg).to(cfg.device)

print(f"Using Device : {cfg.device}")

# Verify GPT-style weight tying between the token embedding and LM head.
embedding_weight = model.token_embedding.embedding.weight
lm_head_weight = model.lm_head.weight

weights_are_shared = embedding_weight is lm_head_weight
storage_is_shared = embedding_weight.data_ptr() == lm_head_weight.data_ptr()

print(f"Weight objects shared : {weights_are_shared}")
print(f"Weight storage shared : {storage_is_shared}")

assert weights_are_shared and storage_is_shared, (
    "Token embedding and LM head are not sharing weights"
)


# --------------------------------------------------
# Optimizer & Loss
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.learning_rate
)

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# Checkpoint Directory
# --------------------------------------------------

os.makedirs("checkpoints", exist_ok=True)


# --------------------------------------------------
# Training Loop
# --------------------------------------------------

# --------------------------------------------------
# Training Loop
# --------------------------------------------------

for epoch in range(cfg.epochs):

    model.train()

    total_loss = 0.0

    epoch_start = time.time()

    for batch_idx, (x, y) in enumerate(loader):

        batch_start = time.time()

        x = x.to(cfg.device)
        y = y.to(cfg.device)

        optimizer.zero_grad()

        # ----------------------------
        # Forward
        # ----------------------------
        forward_start = time.time()

        logits = model(x)

        loss = criterion(
            logits.view(-1, cfg.vocab_size),
            y.view(-1)
        )

        forward_end = time.time()

        # ----------------------------
        # Backward
        # ----------------------------
        backward_start = time.time()

        loss.backward()

        optimizer.step()

        backward_end = time.time()

        total_loss += loss.item()

        batch_end = time.time()

        # Print only first 5 batches
        if batch_idx < 5:

            print(
                f"\nBatch {batch_idx + 1}"
            )

            print(
                f"Forward : {forward_end-forward_start:.4f}s"
            )

            print(
                f"Backward: {backward_end-backward_start:.4f}s"
            )

            print(
                f"Total    : {batch_end-batch_start:.4f}s"
            )

    epoch_end = time.time()

    avg_loss = total_loss / len(loader)

    print(
        f"\nEpoch [{epoch + 1}/{cfg.epochs}] "
        f"Loss: {avg_loss:.4f}"
    )

    print(
        f"Epoch Time : {(epoch_end-epoch_start)/60:.2f} minutes"
    )

    checkpoint_path = f"checkpoints/minigpt_epoch_{epoch+1}.pth"

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved -> {checkpoint_path}")

print("\nTraining Complete!")

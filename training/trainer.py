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

import config

from model.gpt import MiniGPT
from training.dataset import TextDataset


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
    context_length=config.MAX_SEQ_LEN
)


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)


# --------------------------------------------------
# Create Model
# --------------------------------------------------

model = MiniGPT().to(config.DEVICE)

print(f"Using Device : {config.DEVICE}")


# --------------------------------------------------
# Optimizer & Loss
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE
)

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# Checkpoint Directory
# --------------------------------------------------

os.makedirs("checkpoints", exist_ok=True)


# --------------------------------------------------
# Training Loop
# --------------------------------------------------

for epoch in range(config.EPOCHS):

    model.train()

    total_loss = 0.0

    for x, y in loader:

        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(x)

        # Reshape for CrossEntropyLoss
        loss = criterion(
            logits.view(-1, config.VOCAB_SIZE),
            y.view(-1)
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch [{epoch + 1}/{config.EPOCHS}] "
        f"Loss: {avg_loss:.4f}"
    )

    # Save checkpoint after every epoch
    checkpoint_path = f"checkpoints/minigpt_epoch_{epoch + 1}.pth"

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
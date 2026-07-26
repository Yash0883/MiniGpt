"""Train MiniGPT with clipping, warmup, validation, and resumable checkpoints."""

import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer

from config import GPTConfig
from model.gpt import MiniGPT
from training.dataset import TextDataset


cfg = GPTConfig()


def cosine_warmup_lambda(step, warmup_steps, total_steps, min_lr_ratio):
    """Learning-rate multiplier for linear warmup followed by cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)

    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = torch.zeros((), device=cfg.device)

    for x, y in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        logits = model(x)
        loss = criterion(
            logits.reshape(-1, cfg.vocab_size),
            y.reshape(-1),
        )
        total_loss += loss.detach()

    return (total_loss / len(loader)).item()


# --------------------------------------------------
# Load and split tokenized text
# --------------------------------------------------

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

load_kwargs = {"split": cfg.dataset_split}
if cfg.dataset_config is not None:
    load_kwargs["name"] = cfg.dataset_config

raw_dataset = load_dataset(cfg.dataset_name, **load_kwargs)
token_ids = []

for index, row in enumerate(raw_dataset):
    if cfg.max_documents is not None and index >= cfg.max_documents:
        break

    text = row.get(cfg.text_column)
    if not text:
        continue

    token_ids.extend(tokenizer.encode(text).ids)

    if cfg.max_train_tokens is not None and len(token_ids) >= cfg.max_train_tokens:
        token_ids = token_ids[:cfg.max_train_tokens]
        break

if len(token_ids) <= cfg.max_seq_len:
    raise ValueError("The selected dataset contains too few tokens")

# The tokenizer vocabulary is the source of truth for the output head size.
cfg.vocab_size = tokenizer.get_vocab_size()

split_index = int(len(token_ids) * (1.0 - cfg.validation_split))
train_token_ids = token_ids[:split_index]
val_token_ids = token_ids[split_index:]

print(f"Dataset      : {cfg.dataset_name}")
print(f"Total Tokens : {len(token_ids)}")
print(f"Vocab Size   : {cfg.vocab_size}")
print(f"Train Tokens : {len(train_token_ids)}")
print(f"Val Tokens   : {len(val_token_ids)}")

train_dataset = TextDataset(train_token_ids, cfg.max_seq_len)
val_dataset = TextDataset(val_token_ids, cfg.max_seq_len)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
)

# --------------------------------------------------
# Model, optimizer, scheduler, and loss
# --------------------------------------------------

model = MiniGPT(cfg).to(cfg.device)
print(f"Using Device : {cfg.device}")

embedding_weight = model.token_embedding.embedding.weight
lm_head_weight = model.lm_head.weight
weights_are_shared = embedding_weight is lm_head_weight
storage_is_shared = embedding_weight.data_ptr() == lm_head_weight.data_ptr()
print(f"Weight objects shared : {weights_are_shared}")
print(f"Weight storage shared : {storage_is_shared}")
assert weights_are_shared and storage_is_shared

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
criterion = nn.CrossEntropyLoss()

total_steps = cfg.epochs * len(train_loader)
min_lr_ratio = cfg.min_learning_rate / cfg.learning_rate
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_warmup_lambda(
        step,
        cfg.warmup_steps,
        total_steps,
        min_lr_ratio,
    ),
)

# --------------------------------------------------
# Resume, if requested in GPTConfig
# --------------------------------------------------

start_epoch = 0
best_val_loss = float("inf")

if cfg.resume_from:
    checkpoint = torch.load(cfg.resume_from, map_location=cfg.device)
    model.load_state_dict(
        checkpoint.get("model", checkpoint.get("model_state_dict"))
    )
    optimizer.load_state_dict(
        checkpoint.get("optimizer", checkpoint.get("optimizer_state_dict"))
    )
    scheduler_state = checkpoint.get("scheduler", checkpoint.get("scheduler_state_dict"))
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint.get(
        "best_loss",
        checkpoint.get("best_val_loss", float("inf")),
    )
    print(f"Resumed from epoch {start_epoch}")

# --------------------------------------------------
# Training loop
# --------------------------------------------------

os.makedirs(cfg.checkpoint_dir, exist_ok=True)

for epoch in range(start_epoch, cfg.epochs):
    model.train()
    total_train_loss = torch.zeros((), device=cfg.device)
    epoch_start = time.time()

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(cfg.device)
        y = y.to(cfg.device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(
            logits.reshape(-1, cfg.vocab_size),
            y.reshape(-1),
        )
        loss.backward()

        # Prevent unusually large gradients from destabilizing training.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=cfg.grad_clip,
        )

        optimizer.step()
        scheduler.step()
        total_train_loss += loss.detach()

        if (batch_idx + 1) % cfg.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1}/{cfg.epochs} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss {loss.detach().item():.4f} | "
                f"LR {current_lr:.2e} | "
                f"Grad {float(grad_norm):.2f}",
                flush=True,
            )

    train_loss = (total_train_loss / len(train_loader)).item()
    val_loss = evaluate(model, val_loader, criterion)
    perplexity = math.exp(min(val_loss, 20.0))
    epoch_time = (time.time() - epoch_start) / 60.0
    current_lr = scheduler.get_last_lr()[0]
    is_best = val_loss < best_val_loss

    if is_best:
        best_val_loss = val_loss

    print(
        f"\nEpoch:        {epoch + 1}/{cfg.epochs}\n"
        f"Train Loss:   {train_loss:.4f}\n"
        f"Val Loss:     {val_loss:.4f}\n"
        f"Perplexity:   {perplexity:.2f}\n"
        f"LR:           {current_lr:.2e}\n"
        f"Time:         {epoch_time:.2f} min\n"
        f"Best Model:   {'✓' if is_best else ' '}",
        flush=True,
    )

    checkpoint_path = os.path.join(
        cfg.checkpoint_dir,
        f"minigpt_epoch_{epoch + 1}.pth",
    )
    torch.save(
        {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_val_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": cfg.to_dict(),
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved -> {checkpoint_path}", flush=True)

print("\nTraining Complete!")

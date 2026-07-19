import torch

# ===============================
# DATA
# ===============================

VOCAB_SIZE = 8000
MAX_SEQ_LEN = 128

# ===============================
# MODEL
# ===============================

EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
FFN_HIDDEN = EMBED_DIM * 4
DROPOUT = 0.1

# ===============================
# TRAINING
# ===============================

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 10

# ===============================
# DEVICE
# ===============================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
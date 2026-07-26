import torch
from tokenizers import Tokenizer

from config import GPTConfig
from model.gpt import MiniGPT


cfg = GPTConfig()


tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer.json"
)

# The tokenizer vocabulary must match the model's output vocabulary.
cfg.vocab_size = tokenizer.get_vocab_size()

model = MiniGPT(cfg).to(cfg.device)

checkpoint = torch.load(
    "checkpoints/minigpt_epoch_10.pth",
    map_location=cfg.device
)

model_state = checkpoint.get(
    "model",
    checkpoint.get("model_state_dict"),
)
if model_state is None:
    raise KeyError("Checkpoint does not contain model weights")

model.load_state_dict(model_state)

model.eval()


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    """Sample one token from logits using temperature, top-k, and top-p."""

    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive or None")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be between 0 and 1")

    # temperature=0 is useful when deterministic greedy decoding is desired.
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    # Keep only the k highest-scoring tokens.
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        top_values, _ = torch.topk(logits, k, dim=-1)
        cutoff = top_values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    # Keep the smallest set of tokens whose cumulative probability reaches p.
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove = cumulative_probs > top_p
        # Keep the first token that crosses the probability threshold.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_indices, sorted_logits)

    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def generate(
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
    top_p=None,
):

    token_ids = tokenizer.encode(prompt).ids

    tokens = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=cfg.device
    )

    # RoPE and the KV cache support the configured context length only.
    tokens = tokens[:, -cfg.max_seq_len:]

    with torch.no_grad():

        # Process the prompt once and retain every block's K/V tensors.
        logits, past_key_values = model(tokens, use_cache=True)

        for _ in range(max_new_tokens):
            next_token = sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

            if tokens.size(1) >= cfg.max_seq_len:
                # Once the cache is full, slide the context window and rebuild
                # the cache for the retained tokens.
                tokens = tokens[:, -cfg.max_seq_len:]
                logits, past_key_values = model(tokens, use_cache=True)
            else:
                # Only the newly generated token passes through the model.
                logits, past_key_values = model(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

    generated_ids = tokens[0].tolist()

    return tokenizer.decode(generated_ids)


if __name__ == "__main__":

    prompt = "The little girl:"

    print(
        generate(
            prompt,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
        )
    )

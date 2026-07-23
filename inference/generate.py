import torch
from tokenizers import Tokenizer

from config import GPTConfig
from model.gpt import MiniGPT


cfg = GPTConfig()


tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer.json"
)

model = MiniGPT(cfg).to(cfg.device)

checkpoint = torch.load(
    "checkpoints/minigpt_epoch_10.pth",
    map_location=cfg.device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


def generate(prompt, max_new_tokens=100):

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
            next_token_logits = logits[:, -1, :]

            probs = torch.softmax(
                next_token_logits,
                dim=-1
            )

            next_token = torch.argmax(
                probs,
                dim=-1,
                keepdim=True
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

    prompt = "LORD:"

    print(generate(prompt))

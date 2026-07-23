import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig


class RotaryEmbedding(nn.Module):
    """Apply rotary positional information to query and key vectors."""

    def __init__(self, head_dim, max_seq_len, base=10_000):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")

        frequencies = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inverse_frequency = 1.0 / (base ** (frequencies / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequency)

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, position_offset=0):
        # x has shape (batch, heads, sequence, head_dim).
        seq_len = x.size(-2)
        end_position = position_offset + seq_len
        if end_position > self.cos.size(0):
            raise ValueError(
                f"Position {end_position} exceeds MAX_SEQ_LEN "
                f"({self.cos.size(0)})"
            )

        cos = self.cos[position_offset:end_position].to(dtype=x.dtype)
        sin = self.sin[position_offset:end_position].to(dtype=x.dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        even = x[..., 0::2]
        odd = x[..., 1::2]

        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos),
            dim=-1,
        )

        return rotated.flatten(-2)


class MultiHeadAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        embed_dim = config.embed_dim
        assert embed_dim % config.num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = config.num_heads
        self.head_dim = embed_dim // self.num_heads

        self.rotary = RotaryEmbedding(
            self.head_dim,
            config.max_seq_len,
        )

        # One projection for QKV
        self.qkv = nn.Linear(
            embed_dim,
            embed_dim * 3,
            bias=False
        )

        # Output projection
        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

    def forward(self, x, past_key_value=None, position_offset=0, use_cache=False):

        B, T, C = x.shape

        # -----------------------------
        # QKV Projection
        # -----------------------------
        qkv = self.qkv(x)

        # (B,T,3C) → (B,T,C),(B,T,C),(B,T,C)
        Q, K, V = qkv.chunk(3, dim=-1)

        # -----------------------------
        # Split Heads
        # -----------------------------
        Q = Q.view(B, T, self.num_heads, self.head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim)

        # (B,T,H,D) → (B,H,T,D)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Rotate Q and K according to their token positions. V is not rotated.
        Q = self.rotary(Q, position_offset=position_offset)
        K = self.rotary(K, position_offset=position_offset)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            K = torch.cat((past_key, K), dim=-2)
            V = torch.cat((past_value, V), dim=-2)

        # Scaled dot-product attention applies the 1/sqrt(head_dim) scaling,
        # causal masking, and softmax internally.
        if past_key_value is None:
            output = F.scaled_dot_product_attention(
                Q, K, V, is_causal=True, dropout_p=0.0
            )
        else:
            # With cached keys, the new query tokens may attend to all cached
            # positions and to the new tokens up to their current position.
            query_len = Q.size(-2)
            key_len = K.size(-2)
            past_len = key_len - query_len
            query_positions = torch.arange(
                past_len,
                past_len + query_len,
                device=x.device,
            )[:, None]
            key_positions = torch.arange(key_len, device=x.device)[None, :]
            attention_mask = key_positions <= query_positions

            output = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=attention_mask,
                is_causal=False,
                dropout_p=0.0,
            )

        # -----------------------------
        # Merge Heads
        # -----------------------------
        output = output.transpose(1, 2)

        output = output.contiguous().view(
            B,
            T,
            C
        )

        output = self.out_proj(output)

        if use_cache:
            return output, (K, V)

        return output

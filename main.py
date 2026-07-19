# # import torch
# # from model.embeddings import TokenEmbedding
# # from model.attention import SelfAttention

# # VOCAB_SIZE = 10000
# # EMBED_DIM = 512

# # embedding = TokenEmbedding(
# #     VOCAB_SIZE,
# #     EMBED_DIM
# # )

# # # attention = SelfAttention(EMBED_DIM)

# # # tokens = torch.tensor([1, 5, 42, 700])

# # # output = embedding(tokens)
# # # Q, K, V = attention(output)

# # #print(output.shape)
# # #print(output[0])

# # # print("Q:", Q.shape)
# # # print("K:", K.shape)
# # # print("V:", V.shape)

# # attention = SelfAttention(512)

# # x = torch.randn(2, 5, 512)

# # Q, K, V = attention(x)

# # print(Q.shape)
# # print(K.shape)
# # print(V.shape)


# import config

# model = MiniGPT().to(config.DEVICE)

# dummy = torch.randint(
#     0,
#     config.VOCAB_SIZE,
#     (2, 10),
#     device=config.DEVICE
# )

# out = model(dummy)

# print(out.shape)




# import torch
# from model.feedforward import FeedForward

# ffn = FeedForward(512)
# x = torch.randn(2,5,512)
# out = ffn(x)

# print(out.shape)


import torch
import config

from model.gpt import MiniGPT

model = MiniGPT().to(config.DEVICE)

dummy = torch.randint(
    0,
    config.VOCAB_SIZE,
    (2, 10),
    device = config.DEVICE
)

out = model(dummy) 

print("Input:", dummy.shape)
print("Output:", out.shape)


# from model.transformer_block import TransformerBlock

# block = TransformerBlock(embed_dim=512)

# x = torch.randn(2, 5, 512)

# out = block(x)

# print(out.shape)
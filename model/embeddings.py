
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
    
    def forward(self, token_ids):
        return self.embedding(token_ids)



class PositionalEmbedding(nn.Module):

    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, embed_dim)
        
    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(
            seq_len,
            device=token_ids.device

        )

        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
            
        return self.embedding(positions)
    

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from time import time


class FourierEmbedding(nn.Module):
    def __init__(self, embedding_dim=64, max_freq=1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.freqs = torch.exp(torch.linspace(0, torch.log(torch.tensor(max_freq)), embedding_dim // 2))

    def forward(self, t):
        """
        Compute the Fourier embedding for a given time step t.
        
        :param t: Tensor of shape (batch_size, 1) containing time steps.
        :return: Fourier features of shape (batch_size, embedding_dim).
        """
        t = t.unsqueeze(-1)  # Ensure shape is (batch_size, 1)
        scaled_t = t * self.freqs.to(t.device)  # Apply frequencies
        embeddings = torch.cat([torch.sin(scaled_t), torch.cos(scaled_t)], dim=-1)  # Concatenate sin and cos
        return embeddings.reshape(t.shape[0], self.embedding_dim)

# The score network itself

class ScoreNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, time_embed_dim=32, a=0, b=[0, 2], c=0, **kwargs):
        super().__init__()
#        self.time_mlp = nn.Sequential(
#            nn.Linear(1, time_embed_dim),
#            nn.SiLU(),
#            nn.Linear(time_embed_dim, time_embed_dim),
#            nn.SiLU()
#        )

        self.time_mlp = FourierEmbedding(embedding_dim=time_embed_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.a = torch.tensor(a, dtype=torch.float)
        self.b = torch.tensor(b, dtype=torch.float)
        self.c = torch.tensor(c, dtype=torch.float)


    def forward(self, x, t):
        t_embedding = self.time_mlp(t)
        xt = torch.cat([x, t_embedding], dim=-1)
        return self.net(xt)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#https://www.kaggle.com/code/aisuko/causal-self-attention#Self-Attention-Mechanism(Scaled-dot-product-attention)

class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.Q = nn.Linear(h_dim, h_dim)
        self.K = nn.Linear(h_dim, h_dim)
        self.V = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.atten_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((self.max_T, self.max_T))
        mask = torch.tril(ones).view(1, 1, self.max_T, self.max_T)

        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape

        N, D = self.n_heads, C//self.n_heads  #N=num of heads D=attention dim

        #rearrange Q, K, V as (B,T,N,D)
        Q = self.Q(x).view(B, T, N, D).transpose(1, 2)
        K = self.K(x).view(B, T, N, D).transpose(1, 2)
        V = self.V(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = Q @ K.transpose(2, 3) / math.sqrt(D)

        #casul mask applied to weights
        weights = weights.masked_fill(self.mask[..., :T, :T]==0, float('-inf')) 

        #normalized weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        #attention (B, N, T, D)
        attention = self.atten_drop(normalized_weights @ V)

        #gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out
    

class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4*h_dim),
            nn.GELU(),
            nn.Linear(4*h_dim, h_dim),
            nn.Dropout(drop_p)
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        x = x + self.attention(x)
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x
    

    
import torch
from torch import nn
from torch.nn import functional as F

class CasualSelfAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embed % n_head == 0
        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.n_head = n_head
        self.n_embd = n_embed
        self.bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y # (B,T,C)

class MLP(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, n_embed * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(n_embed * 4, n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x # (B,T,C)

class Block(nn.Module):
    def __init__(self, n_embed: int, n_head: int, block_size: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = CasualSelfAttention(n_embed, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x # (B,T,C)

class Transformer(nn.Module):
    def __init__(self, n_embed: int, n_head: int, block_size: int, n_block: int, n_pred: int):
        super().__init__()
        self.n_pred = n_pred
        self.transformer = nn.ModuleDict(dict(
            h=nn.ModuleList([Block(n_embed, n_head, block_size) for _ in range(n_block)]),
            ln_f=nn.LayerNorm(n_embed)
        ))
        self.lm_head = nn.Linear(n_embed, n_pred, bias=False)

        # init parms
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        x = idx  # Directly use the input data
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        # logits = self.lm_head(x)  # (B, T, n_pred)
        logits = x[:, -self.n_pred:, :]  # (B, n_pred, C) -> Taking the last n_pred predictions

        logits = logits.squeeze(-1)  # (B, n_pred)

        loss = None
        if target is not None:
            target = target[:, -self.n_pred:]  # Ensure target matches the shape of logits
            loss = F.mse_loss(logits, target)  # Use MSE loss for regression
        return logits, loss

    @staticmethod
    def accuracy(logits, targets):
        # Accuracy is not a typical metric for regression tasks; we can use R2 score or MAE instead
        mae = torch.mean(torch.abs(logits - targets))
        return mae


model = Transformer(n_embed=5, n_head=5, block_size=70, n_block=4, n_pred=10)
batch = torch.rand((32, 70, 5))
output, _ = model(batch)
print(output.shape)



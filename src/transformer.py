from typing import Tuple
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, embd_size, block_size, head_size, dropout):
        super(Head, self).__init__()
        self.key = nn.Linear(embd_size, head_size, bias=False)
        self.query = nn.Linear(embd_size, head_size, bias=False)
        self.value = nn.Linear(embd_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        wei = query @ key.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        x = wei @ value
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embd_size, block_size, head_size, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [Head(embd_size, block_size, head_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embd_size, embd_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, embd_size, dropout, dim_scaler=4):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_size, embd_size * dim_scaler),
            nn.ReLU(),
            nn.Linear(embd_size * dim_scaler, embd_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embd_size, block_size, num_heads, dropout):
        super(Block, self).__init__()
        head_size = embd_size // num_heads
        self.sa = MultiHeadAttention(
            embd_size, block_size, head_size, num_heads, dropout
        )
        self.ffwd = FeedForward(embd_size, dropout)
        self.ln1 = nn.LayerNorm(embd_size)
        self.ln2 = nn.LayerNorm(embd_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embd_size: int,
        block_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super(Transformer, self).__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embd_size)
        self.position_embeddig = nn.Embedding(block_size, embd_size)
        self.blocks = nn.Sequential(
            *[
                Block(embd_size, block_size, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(embd_size)
        self.lm_head = nn.Linear(embd_size, vocab_size)

    def forward(self, x: Tensor, y: Tensor = None) -> Tuple[Tensor, Tensor]:
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embeddig(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if y is not None:
            logits = logits.view(-1, logits.shape[-1])
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)
        else:
            loss = None
        return logits, loss

    def generate(self, max_new_tokens, idx: Tensor = None):
        device = next(self.parameters()).device
        if idx is None:
            idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        else:
            idx = idx.to(device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

    def train_model(self, data, optim, lr, epochs, log_every=500, eval_iters=100):
        device = next(self.parameters()).device
        optimizer_class = getattr(torch.optim, optim)
        optimizer = optimizer_class(self.parameters(), lr=lr)
        for epoch in range(epochs):
            xb, yb = data.get_batch(data.train_data)
            xb, yb = xb.to(device), yb.to(device)
            _, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if log_every and (epoch + 1) % log_every == 0:
                train_loss, valid_loss = self.estimate_loss(data, eval_iters)
                print(
                    f"epoch {epoch + 1}/{epochs}: "
                    f"train loss {train_loss:.4f}, valid loss {valid_loss:.4f}"
                )

    @torch.no_grad()
    def estimate_loss(self, data, eval_iters):
        device = next(self.parameters()).device
        out = []
        self.eval()
        for sub_data in [data.train_data, data.valid_data]:
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                X, Y = data.get_batch(sub_data)
                X, Y = X.to(device), Y.to(device)
                _, loss = self(X, Y)
                losses[i] = loss.item()
            out.append(losses.mean())
        self.train()
        return out

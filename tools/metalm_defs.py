"""Model classes shared by the Meta LM scripts (re-implementation of encodec 0.1.1)."""
import torch
from torch import nn


def create_sin_embedding(positions, dim, max_period=10000):
    half = dim // 2
    adim = torch.arange(half).view(1, 1, -1)
    phase = positions / (max_period ** (adim / (half - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


class Layer(nn.TransformerEncoderLayer):
    """Post-norm encoder layer attending to [past state, current] within `past_context`."""

    def forward(self, x, x_past, past_context):
        sa_input = x
        t, h = x.shape[1], x_past.shape[1]
        keys = torch.cat([x_past, x], dim=1)
        delta = torch.arange(h, t + h).view(-1, 1) - torch.arange(t + h).view(1, -1)
        valid = (delta >= 0) & (delta <= past_context)
        att = self.self_attn(x, keys, keys, attn_mask=~valid, need_weights=False)[0]
        x = self.norm1(x + att)
        x = self.norm2(x + self.linear2(self.activation(self.linear1(x))))
        return x, sa_input


class Transformer(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=5, past_context=1000):
        super().__init__()
        self.past_context = past_context
        self.norm_in = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            Layer(dim, num_heads, dim * 4, activation=nn.functional.gelu, batch_first=True,
                  dropout=0.0) for _ in range(num_layers)])

    def forward(self, x, states=None, offset=0):
        t, c = x.shape[1], x.shape[2]
        if states is None:
            states = [torch.zeros_like(x[:, :1]) for _ in self.layers]
        pos = torch.arange(t).view(1, -1, 1) + offset
        x = self.norm_in(x) + create_sin_embedding(pos, c)
        new_state = []
        for st, layer in zip(states, self.layers):
            x, inp = layer(x, st, self.past_context)
            new_state.append(torch.cat([st, inp], dim=1)[:, -self.past_context:, :])
        return x, new_state, offset + t


class LMModel(nn.Module):
    def __init__(self, n_q=32, card=1024, dim=200, **kwargs):
        super().__init__()
        self.card, self.n_q, self.dim = card, n_q, dim
        self.transformer = Transformer(dim=dim, **kwargs)
        self.emb = nn.ModuleList([nn.Embedding(card + 1, dim) for _ in range(n_q)])
        self.linears = nn.ModuleList([nn.Linear(dim, card) for _ in range(n_q)])

    def forward(self, indices, states=None, offset=0):
        _, k, _ = indices.shape
        input_ = sum([self.emb[i](indices[:, i]) for i in range(k)])
        out, states, offset = self.transformer(input_, states, offset)
        logits = torch.stack([self.linears[i](out) for i in range(k)], dim=1).permute(0, 3, 1, 2)
        return torch.softmax(logits, dim=1), states, offset

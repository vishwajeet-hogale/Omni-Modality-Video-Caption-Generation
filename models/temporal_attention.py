import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class TemporalAttention(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(in_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: Optional[torch.Tensor], mask: Optional[torch.Tensor]):
        # query:  [B, in_dim]
        # memory: [B, Tm, H] or None
        # mask:   [B, Tm] (1=valid)
        if memory is None:
            return self.out_proj(self.q_proj(query))  # cold start

        B, Tm, H = memory.shape
        q = self.q_proj(query).unsqueeze(1)  # [B,1,H]
        k = self.k_proj(memory)              # [B,Tm,H]
        v = self.v_proj(memory)              # [B,Tm,H]

        def split_heads(x):                  # [B,L,H] -> [B,h,L,d]
            return x.view(B, x.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)                   # [B,h,1,d]
        k = split_heads(k)                   # [B,h,Tm,d]
        v = split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,h,1,Tm]
        if mask is not None:
            bad = (mask.unsqueeze(1).unsqueeze(1) == 0)                          # [B,1,1,Tm]
            scores = scores.masked_fill(bad, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v)                                              # [B,h,1,d]
        ctx = ctx.transpose(1, 2).contiguous().view(B, 1, self.hidden_dim)       # [B,1,H]
        return self.out_proj(ctx.squeeze(1))                                     # [B,H]


class TemporalMemory:
    def __init__(self, max_len: int = 10):
        self.max_len = max_len
        self.buffer = []  # list of Tensor [B, H]

    def clear(self):
        self.buffer.clear()

    def add(self, cls_proj):  # cls_proj: [B, H]
        self.buffer.append(cls_proj.detach())
        if len(self.buffer) > self.max_len:
            self.buffer.pop(0)

    def get_previous(self, batch_size: int, device):
        if not self.buffer:
            return None, None
        # stack -> [Tm, B, H] then permute to [B, Tm, H]
        mem = torch.stack(self.buffer, dim=0).to(device).permute(1, 0, 2).contiguous()
        mask = torch.ones(mem.size(0), mem.size(1), dtype=torch.long, device=device)  # [B, Tm]
        return mem, mask

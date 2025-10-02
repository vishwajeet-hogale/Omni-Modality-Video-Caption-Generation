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

        def split_heads(x):
            # x: [B, seq_len, hidden_dim]
            B, seq_len, hidden_dim = x.size()
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
        self.expected_dim = None  # Track expected dimension

    def clear(self):
        self.buffer.clear()
        self.expected_dim = None

    def add(self, cls_proj):  # cls_proj: [B, H]
        cls_proj = cls_proj.detach()
        
        # Set expected dimension on first add
        if self.expected_dim is None:
            self.expected_dim = cls_proj.size(-1)
        
        # Validate dimension consistency
        if cls_proj.size(-1) != self.expected_dim:
            print(f"WARNING: Dimension mismatch in temporal memory! Got {cls_proj.size(-1)}, expected {self.expected_dim}")
            # Skip this addition to avoid corrupting memory
            return
            
        self.buffer.append(cls_proj)
        if len(self.buffer) > self.max_len:
            self.buffer.pop(0)

    def get_previous(self, batch_size: int, device):
        if not self.buffer:
            return None, None
        # stack -> [Tm, B, H] then permute to [B, Tm, H]
        mem = torch.stack(self.buffer, dim=0).to(device).permute(1, 0, 2).contiguous()
        
        # Handle batch size mismatch by taking only the needed samples
        if mem.size(0) != batch_size:
            if batch_size < mem.size(0):
                # Take only the first batch_size samples
                mem = mem[:batch_size]
            else:
                # This shouldn't happen normally, but handle gracefully
                return None, None
        
        mask = torch.ones(mem.size(0), mem.size(1), dtype=torch.long, device=device)  # [B, Tm]
        return mem, mask


class EnhancedTemporalMemory:
    """Enhanced temporal memory that stores both multimodal and text features."""
    def __init__(self, max_len: int = 10, text_dim: int = 768):
        self.max_len = max_len
        self.mm_buffer = []      # Multimodal features
        self.text_buffer = []    # Text features
        self.text_dim = text_dim
        
    def clear(self):
        self.mm_buffer.clear()
        self.text_buffer.clear()
        
    def add(self, mm_features, text_features=None):
        # Ensure we have consistent batch sizes
        batch_size = mm_features.shape[0]
        
        # Add multimodal features
        self.mm_buffer.append(mm_features.detach())
        if len(self.mm_buffer) > self.max_len:
            self.mm_buffer.pop(0)
            
        # Add text features (or placeholder)
        if text_features is not None:
            # Ensure text features have the same batch size
            if text_features.shape[0] != batch_size:
                # If batch sizes don't match, take only the first batch_size elements
                text_features = text_features[:batch_size]
            self.text_buffer.append(text_features.detach())
        else:
            # Add zero placeholder for consistency
            device = mm_features.device
            zero_text = torch.zeros(batch_size, self.text_dim, device=device)
            self.text_buffer.append(zero_text)
            
        if len(self.text_buffer) > self.max_len:
            self.text_buffer.pop(0)
    
    def get_previous(self, batch_size: int, device):
        if not self.mm_buffer:
            return None, None, None
            
        # Handle batch size mismatches by ensuring all buffers have the same batch size
        # Take the minimum batch size across all buffers to avoid mismatches
        min_batch_size = min(batch_size, min(buf.size(0) for buf in self.mm_buffer))
        
        # If the requested batch size is larger than what we have, return None
        if min_batch_size < batch_size:
            return None, None, None
            
        # Stack memories with consistent batch size
        # Each buffer should be [batch_size, feature_dim], stack along sequence dimension
        mm_memory = torch.stack([buf[:min_batch_size] for buf in self.mm_buffer], dim=1)  # [B, T, feature_dim]
        text_memory = torch.stack([buf[:min_batch_size] for buf in self.text_buffer], dim=1)  # [B, T, feature_dim]
        
        # Create attention mask
        seq_len = len(self.mm_buffer)
        mask = torch.ones(min_batch_size, seq_len, device=device)
        
        return mm_memory, text_memory, mask


class EnhancedTemporalAttention(nn.Module):
    """Enhanced temporal attention that handles both multimodal and text features."""
    def __init__(self, multimodal_dim: int, text_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Separate attention for multimodal and text
        self.mm_attention = TemporalAttention(multimodal_dim, hidden_dim, num_heads, dropout)
        self.text_attention = TemporalAttention(text_dim, hidden_dim, num_heads, dropout)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(multimodal_dim + 2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, current_mm, current_text, mm_memory, text_memory, mask):
        # Multimodal temporal attention
        mm_context = self.mm_attention(current_mm, mm_memory, mask)
        
        # Text temporal attention (if text features available)
        text_context = torch.zeros_like(mm_context)
        if current_text is not None and text_memory is not None:
            text_context = self.text_attention(current_text, text_memory, mask)
        
        # Combine all features
        combined = torch.cat([current_mm, mm_context, text_context], dim=-1)
        return self.fusion(combined)

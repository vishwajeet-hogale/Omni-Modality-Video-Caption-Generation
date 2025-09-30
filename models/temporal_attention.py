import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for video captioning.
    Allows current frame to attend to previous frames in the sequence.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, current_features: torch.Tensor, 
                previous_features: List[torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply temporal attention to current frame using previous frames.
        
        Args:
            current_features: [batch_size, hidden_dim] - Current frame features
            previous_features: List of [batch_size, hidden_dim] - Previous frame features
            attention_mask: [batch_size, seq_len] - Optional attention mask
            
        Returns:
            [batch_size, hidden_dim] - Attended features
        """
        if not previous_features:
            return current_features
            
        batch_size = current_features.shape[0]
        
        # Stack previous features: [batch_size, seq_len, hidden_dim]
        prev_stack = torch.stack(previous_features, dim=1)
        seq_len = prev_stack.shape[1]
        
        # Project to Q, K, V
        Q = self.query_proj(current_features).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        K = self.key_proj(prev_stack)  # [batch_size, seq_len, hidden_dim]
        V = self.value_proj(prev_stack)  # [batch_size, seq_len, hidden_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, 1, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, 1, seq_len]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores.masked_fill_(attention_mask.unsqueeze(1).unsqueeze(1) == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # [batch_size, num_heads, 1, head_dim]
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_dim)
        attended = attended.squeeze(1)  # [batch_size, hidden_dim]
        
        # Output projection and residual connection
        output = self.out_proj(attended)
        output = self.layer_norm(output + current_features)
        
        return output

class TemporalMemory(nn.Module):
    """
    Memory buffer to store previous frame features for temporal attention.
    """
    def __init__(self, max_length: int = 10):
        super().__init__()
        self.max_length = max_length
        self.memory_buffer = []
        
    def add(self, features: torch.Tensor):
        """Add new features to memory buffer"""
        self.memory_buffer.append(features.detach())
        if len(self.memory_buffer) > self.max_length:
            self.memory_buffer.pop(0)
            
    def get_previous(self) -> List[torch.Tensor]:
        """Get previous features for temporal attention"""
        return self.memory_buffer.copy()
    
    def clear(self):
        """Clear memory buffer"""
        self.memory_buffer.clear()
        
    def __len__(self):
        return len(self.memory_buffer)

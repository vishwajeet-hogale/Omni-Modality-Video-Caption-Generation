import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class AudioTokenizer(nn.Module):
    """Convert audio to tokens for ViT processing"""
    def __init__(self, config):
        super().__init__()
        self.audio_dim = config.audio_dim
        self.hidden_dim = config.hidden_dim
        self.patch_size = config.patch_size
        self.kernel_size = config.kernel_size
        
        # Convert audio to spectrogram
        self.spectrogram = nn.Conv1d(1, self.audio_dim, kernel_size=self.kernel_size)
        
        # Convert spectrogram to patches (like image patches)
        self.patch_embed = nn.Conv2d(1, self.hidden_dim, 
                                   kernel_size=self.patch_size, 
                                   stride=self.patch_size)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, self.hidden_dim))
        
    def forward(self, audio):
        # audio: [batch_size, seq_len]
        batch_size = audio.size(0)
        
        # Convert to spectrogram
        spec = self.spectrogram(audio.unsqueeze(1))  # [batch_size, audio_dim, seq_len]
        spec = spec.unsqueeze(1)  # [batch_size, 1, audio_dim, seq_len]
        
        # Convert to patches
        patches = self.patch_embed(spec)  # [batch_size, hidden_dim, height, width]
        patches = patches.flatten(2).transpose(1, 2)  # [batch_size, num_patches, hidden_dim]
        
        # Add positional encoding
        seq_len = patches.size(1)
        if seq_len <= self.pos_embed.size(1):
            patches = patches + self.pos_embed[:, :seq_len, :]
        
        return patches

class TextTokenizer(nn.Module):
    """Convert text to tokens for ViT processing"""
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.max_length = config.max_length
        
        # Use pre-trained tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, self.hidden_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_length, self.hidden_dim))
        
    def forward(self, text):
        # Tokenize text
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, 
                               truncation=True, max_length=self.max_length)
        token_ids = tokens['input_ids'].to(next(self.parameters()).device)
        
        # Embed tokens
        embeddings = self.embedding(token_ids)
        
        # Add positional encoding
        seq_len = embeddings.size(1)
        embeddings = embeddings + self.pos_embed[:, :seq_len, :]
        
        return embeddings

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, query, key, value):
        # Apply attention
        attended, _ = self.attention(query, key, value)
        
        # Residual connection and normalization
        output = self.norm(query + attended)
        
        return output

class ViTEncoder(nn.Module):
    """Vision Transformer encoder"""
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, self.hidden_dim, kernel_size=16, stride=16)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, self.hidden_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(self.hidden_dim, self.num_heads, batch_first=True)
            for _ in range(self.num_layers)
        ])
        
    def forward(self, x):
        # x: [batch_size, 3, height, width]
        batch_size = x.size(0)
        
        # Convert to patches
        patches = self.patch_embed(x)  # [batch_size, hidden_dim, height, width]
        patches = patches.flatten(2).transpose(1, 2)  # [batch_size, num_patches, hidden_dim]
        
        # Add positional encoding
        seq_len = patches.size(1)
        if seq_len <= self.pos_embed.size(1):
            patches = patches + self.pos_embed[:, :seq_len, :]
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            patches = layer(patches)
        
        return patches

class EnhancedMultiModalViT(nn.Module):
    """Enhanced Multi-Modal ViT for processing multiple modalities"""
    def __init__(self, config):
        super().__init__()
        
        # Modality encoders
        self.image_encoder = ViTEncoder(config.image)
        self.audio_encoder = AudioTokenizer(config.audio)
        self.text_encoder = TextTokenizer(config.text)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(config.hidden_dim, config.num_heads)
        
        # Temporal attention (for video sequences)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(config.hidden_dim, config.num_heads, batch_first=True)
            for _ in range(config.num_fusion_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        
    def forward(self, images, audio, text=None, timestamps=None):
        # Encode each modality
        img_features = self.image_encoder(images)
        aud_features = self.audio_encoder(audio)
        
        # Cross-modal attention
        img_attended = self.cross_attention(img_features, aud_features, aud_features)
        aud_attended = self.cross_attention(aud_features, img_features, img_features)
        
        # Combine features
        combined = torch.cat([img_attended, aud_attended], dim=1)
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            combined = layer(combined)
        
        # Output projection
        output = self.output_proj(combined)
        
        return output

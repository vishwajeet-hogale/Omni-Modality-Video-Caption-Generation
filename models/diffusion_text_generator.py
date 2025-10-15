import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class DiffusionTextGenerator(nn.Module):
    """Diffusion-based text generation model"""
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim
        self.num_timesteps = config.num_timesteps
        self.max_length = config.max_length
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Embedding layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.timestep_embedding = nn.Embedding(self.num_timesteps, self.hidden_dim)
        self.position_embedding = nn.Embedding(self.max_length, self.hidden_dim)
        
        # Transformer for denoising
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self.hidden_dim, nhead=8, batch_first=True),
            num_layers=6
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size)
        
        # Noise schedule
        self.register_buffer('betas', self._get_beta_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def _get_beta_schedule(self):
        """Get noise schedule for diffusion"""
        return torch.linspace(0.0001, 0.02, self.num_timesteps)
    
    def forward(self, x, image_embeddings, timesteps):
        # x: [batch_size, seq_len] - noisy tokens
        # image_embeddings: [batch_size, seq_len, hidden_dim]
        # timesteps: [batch_size] - diffusion timesteps
        
        batch_size, seq_len = x.size()
        
        # Embed tokens
        token_emb = self.token_embedding(x)  # [batch_size, seq_len, hidden_dim]
        
        # Add timestep embedding
        timestep_emb = self.timestep_embedding(timesteps)  # [batch_size, hidden_dim]
        timestep_emb = timestep_emb.unsqueeze(1).expand(-1, seq_len, -1)
        token_emb = token_emb + timestep_emb
        
        # Add positional embedding
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)
        token_emb = token_emb + pos_emb
        
        # Use image embeddings as memory in transformer
        output = self.transformer(token_emb, image_embeddings)
        
        # Project to vocabulary
        logits = self.output_proj(output)
        
        return logits
    
    def sample(self, image_embeddings, max_length=60, num_beams=4):
        """Sample text from the diffusion model"""
        batch_size = image_embeddings.size(0)
        device = image_embeddings.device
        
        # Start with random noise
        noise = torch.randn(batch_size, max_length, self.vocab_size, device=device)
        
        # Denoise step by step
        for t in range(self.num_timesteps - 1, -1, -1):
            timesteps = torch.full((batch_size,), t, device=device)
            noise = self.forward(noise, image_embeddings, timesteps)
        
        # Convert to tokens
        generated_tokens = torch.argmax(noise, dim=-1)
        
        return generated_tokens
    
    def add_noise(self, x, timesteps):
        """Add noise to clean tokens"""
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod[timesteps])
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1)
        
        noisy_x = sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise
        return noisy_x, noise
    
    def compute_loss(self, x, image_embeddings, timesteps):
        """Compute diffusion loss"""
        # Add noise
        noisy_x, noise = self.add_noise(x, timesteps)
        
        # Predict noise
        predicted_noise = self.forward(noisy_x, image_embeddings, timesteps)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss

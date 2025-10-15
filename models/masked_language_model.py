import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

class MaskedLanguageModel(nn.Module):
    """Masked Language Model for predicting masked tokens"""
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim
        self.max_length = config.max_length
        self.mask_prob = config.mask_prob
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Embedding layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.position_embedding = nn.Embedding(self.max_length, self.hidden_dim)
        
        # Encoder for image features
        self.image_encoder = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Decoder for text generation
        self.text_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self.hidden_dim, nhead=8, batch_first=True),
            num_layers=6
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size)
        
    def forward(self, tokens, image_embeddings, mask_positions):
        # tokens: [batch_size, seq_len] - input tokens
        # image_embeddings: [batch_size, seq_len, hidden_dim]
        # mask_positions: [batch_size, num_masked] - positions to mask
        
        batch_size, seq_len = tokens.size()
        
        # Embed tokens
        token_emb = self.token_embedding(tokens)
        
        # Add positional embedding
        pos_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)
        token_emb = token_emb + pos_emb
        
        # Encode image features
        image_features = self.image_encoder(image_embeddings)
        
        # Decode text with image context
        output = self.text_decoder(token_emb, image_features)
        
        # Get predictions for masked positions
        masked_output = output[mask_positions]
        
        return self.output_proj(masked_output)
    
    def create_masked_tokens(self, tokens, mask_prob=0.15):
        """Create masked version of input tokens"""
        batch_size, seq_len = tokens.size()
        device = tokens.device
        
        # Create mask
        mask = torch.rand(batch_size, seq_len, device=device) < mask_prob
        
        # Create masked tokens
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.tokenizer.mask_token_id
        
        return masked_tokens, mask
    
    def compute_loss(self, tokens, image_embeddings, mask_prob=0.15):
        """Compute masked language model loss"""
        # Create masked tokens
        masked_tokens, mask_positions = self.create_masked_tokens(tokens, mask_prob)
        
        # Get predictions
        predictions = self.forward(masked_tokens, image_embeddings, mask_positions)
        
        # Get target tokens
        target_tokens = tokens[mask_positions]
        
        # Compute loss
        loss = F.cross_entropy(predictions, target_tokens)
        
        return loss, predictions, mask_positions

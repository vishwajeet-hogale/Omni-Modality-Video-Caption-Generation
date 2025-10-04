import torch
import torch.nn as nn
from models.encoder import VisionEncoder
from models.decoder import TextDecoder
from models.audio_encoder import AudioEncoder
from models.temporal_attention import TemporalAttention, TemporalMemory, EnhancedTemporalMemory, EnhancedTemporalAttention
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

class SwinBart(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Init encoder
        self.encoder = VisionEncoder(
            encoder_name=cfg.vision_encoder_cfg.encoder,
            output_dim=cfg.vision_encoder_cfg.output_dim,
            projection_dim=cfg.decoder_cfg.hidden_dim
        )

        # Init decoder
        self.decoder = TextDecoder(
            decoder_name=cfg.decoder_cfg.decoder
        )
        self.audio_encoder = AudioEncoder(
            encoder_name=cfg.audio_encoder_cfg.encoder,
            n_mfcc=cfg.audio_encoder_cfg.n_mfcc,
            projection_dim=cfg.decoder_cfg.hidden_dim
        )
        
        # Temporal attention components (optional)
        self.use_temporal_attention = getattr(cfg, 'use_temporal_attention', False)
        self.use_text_feedback = getattr(cfg, 'use_text_feedback', False)  # Disabled for vision+audio only
        
        if self.use_temporal_attention:
            # Standard temporal attention without text feedback
            self.temporal_attention = TemporalAttention(
                in_dim=1536,  # Fused feature dimension (vision + audio)
                hidden_dim=cfg.decoder_cfg.hidden_dim,  # 768
                num_heads=getattr(cfg, 'temporal_attention_heads', 8),
                dropout=getattr(cfg, 'temporal_attention_dropout', 0.1)
            )
            self.temporal_memory = TemporalMemory(
                max_len=getattr(cfg, 'temporal_memory_length', 10)
            )
        self.projection_outputs_to_temp = nn.Linear(50265, cfg.decoder_cfg.hidden_dim)
        self.projection = nn.Linear(2304, 1536)
    def forward(self, images, captions, audio, is_new_video=False):
        B, device = images.size(0), images.device

        # Reset temporal memory if starting a new video
        if is_new_video and self.use_temporal_attention:
            self.temporal_memory.clear()

        # 1) encode multimodal features
        img_feats = self.encoder(images)          # [B, vision_dim]
        aud_feats = self.audio_encoder(audio)     # [B, audio_dim]

        # 2) fuse + normalize
        current_mm_features = torch.cat([img_feats, aud_feats], dim=1)   # [B, 1536]
        current_mm_features = F.normalize(current_mm_features, dim=1)

        # 3) apply temporal attention (vision+audio only)
        enhanced_features = current_mm_features
        if self.use_temporal_attention:
            # Standard temporal attention without text feedback
            memory, mask = self.temporal_memory.get_previous(B, device)
            if memory is not None:
                attn_results = self.temporal_attention(query=current_mm_features, memory=memory, mask=mask)
                combined_prev_current = torch.concat([current_mm_features, attn_results], dim=1)
                enhanced_features = self.projection(combined_prev_current)

        # 5) decode this frame
        outputs = self.decoder.forward(enhanced_features, captions)

        # 5) update memory with current frame features (for next frame)
        if self.use_temporal_attention:
            # Store only multimodal features (vision + audio)
            with torch.no_grad():
                projected_features = self.temporal_attention.q_proj(current_mm_features)
                assert projected_features.size(-1) == self.temporal_attention.hidden_dim, \
                    f"Projected features dim {projected_features.size(-1)} != expected {self.temporal_attention.hidden_dim}"
                self.temporal_memory.add(projected_features.detach())

        return outputs
    def generate(self, images, audio, max_length=60, num_beams=4, is_new_video=False, previous_caption=None):
        B, device = images.size(0), images.device
        
        # Reset temporal memory if starting a new video
        if is_new_video and self.use_temporal_attention:
            self.temporal_memory.clear()
            
        # Encode current frame (vision + audio only)
        image_features = self.encoder(images)
        audio_features = self.audio_encoder(audio)
        current_mm_features = torch.cat((image_features, audio_features), dim=1)
        current_mm_features = F.normalize(current_mm_features, dim=1)
        
        # Apply temporal attention
        enhanced_features = current_mm_features
        if self.use_temporal_attention:
            # Standard temporal attention (vision + audio only)
            memory, mask = self.temporal_memory.get_previous(B, device)
            if memory is not None:
                attn_results = self.temporal_attention(query=current_mm_features, memory=memory, mask=mask)
                combined_features = torch.concat([current_mm_features, attn_results], dim=1)
                enhanced_features = self.projection(combined_features)
            
        # Generate caption
        generated_ids = self.decoder.generate(
            image_audio_embeddings=enhanced_features,
            max_length=max_length,
            num_beams=num_beams
        )
        
        # Update memory with current frame for next generation
        if self.use_temporal_attention:
            # Store only multimodal features (vision + audio)
            with torch.no_grad():
                projected_features = self.temporal_attention.q_proj(current_mm_features)
                assert projected_features.size(-1) == self.temporal_attention.hidden_dim, \
                    f"Projected features dim {projected_features.size(-1)} != expected {self.temporal_attention.hidden_dim}"
                self.temporal_memory.add(projected_features.detach())
            
        return generated_ids
    
    def reset_temporal_memory(self):
        """Reset temporal memory for new sequence"""
        if self.use_temporal_attention:
            self.temporal_memory.clear()
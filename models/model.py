import torch
import torch.nn as nn
from models.encoder import VisionEncoder
from models.decoder import TextDecoder
from models.audio_encoder import AudioEncoder
from models.temporal_attention import TemporalAttention, TemporalMemory

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
        if self.use_temporal_attention:
            self.temporal_attention = TemporalAttention(
                in_dim=cfg.decoder_cfg.hidden_dim,
                hidden_dim=cfg.decoder_cfg.hidden_dim ,
                num_heads=getattr(cfg, 'temporal_attention_heads', 8),
                dropout=getattr(cfg, 'temporal_attention_dropout', 0.1)
            )
            self.temporal_memory = TemporalMemory(
                max_len=getattr(cfg, 'temporal_memory_length', 10)
            )
        self.projection_outputs_to_temp = nn.Linear(50265, cfg.decoder_cfg.hidden_dim)
        self.projection = nn.Linear(2304, 1536)
    def forward(self, images, captions, audio):
        B, device = images.size(0), images.device

        # 1) encode
        img_feats = self.encoder(images)          # [B, I]
        aud_feats = self.audio_encoder(audio)     # [B, A]

        # 2) fuse + normalize
        fused = torch.cat([img_feats, aud_feats], dim=1)   # [B, 1536]
        fused = nn.functional.normalize(fused, dim=1)

        # 3) optional temporal attention
        if self.use_temporal_attention:
            memory, mask = self.temporal_memory.get_previous(B, device)

            if memory is None:
                memory = torch.zeros(B, 1, self.temporal_attention.hidden_dim, device=device)  # [B,1,H]
                mask = torch.ones(B, 1, dtype=torch.long, device=device)  # [B,1]

            if not hasattr(self, "input_proj"):
                self.input_proj = nn.Linear(fused.size(1), memory.size(-1)).to(device)

            attn_query = self.input_proj(fused)  # [B,H]
            attn_results = self.temporal_attention(query=attn_query, memory=memory, mask=mask)

        # 4) decode this frame (decoder expects [B,1536])
        combined_prev_current = torch.concat([fused, attn_results], dim=1) if self.use_temporal_attention else fused
        combined_prev_current = self.projection(combined_prev_current) if self.use_temporal_attention else combined_prev_current
        outputs = self.decoder.forward(combined_prev_current, captions)
        logits = outputs.logits

        # 5) update memory only if enabled
        if self.use_temporal_attention:
            cls_logits = logits[:, 0, :]
            cls_proj   = self.projection_outputs_to_temp(cls_logits)
            self.temporal_memory.add(cls_proj)

        return outputs
    def generate(self, images, audio, max_length=128, num_beams=4):
        image_features = self.encoder(images)
        audio_features = self.audio_encoder(audio)
        concat_features = torch.cat((image_features, audio_features), dim=1)
        normalized_features = nn.functional.normalize(concat_features, dim=1)
        generated_ids = self.decoder.generate(
            image_audio_embeddings=normalized_features,
            max_length=max_length,
            num_beams=num_beams
        )
        return generated_ids
    
    def reset_temporal_memory(self):
        """Reset temporal memory for new sequence"""
        if self.use_temporal_attention:
            self.temporal_memory.clear()
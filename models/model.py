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
                hidden_dim=cfg.decoder_cfg.hidden_dim * 2,
                num_heads=getattr(cfg, 'temporal_attention_heads', 8),
                dropout=getattr(cfg, 'temporal_attention_dropout', 0.1)
            )
            self.temporal_memory = TemporalMemory(
                max_length=getattr(cfg, 'temporal_memory_length', 10)
            )

    def forward(self, images, captions, audio):
        image_features = self.encoder(images)
        audio_features = self.audio_encoder(audio)
        
        # Apply temporal attention if enabled
        if self.use_temporal_attention and self.training:
            batch_size = image_features.shape[0]
            temporal_features = []
            
            for i in range(batch_size):
                current_image = image_features[i:i+1]
                current_audio = audio_features[i:i+1]
                
                # Get previous features from memory
                previous_features = self.temporal_memory.get_previous()
                print(f"🔍 TEMPORAL DEBUG:")
                print(f"  📊 Previous features: {len(previous_features)}")
                concat_feat = torch.cat((current_image, current_audio), dim=1)  # 1536 dims
                normalized_feat = nn.functional.normalize(concat_feat, dim=1)
                print(f"  📊 Current features: {normalized_feat.shape}")

                # Apply temporal attention to COMBINED features
                if previous_features:
                    print(f"  🎯 APPLYING TEMPORAL ATTENTION")
                    attended_feat = self.temporal_attention(
                        current_features=normalized_feat,  # 1536 dims
                        previous_features=previous_features  # 1536 dims
                    )
                    print(f"  ✅ Attended features: {attended_feat.shape}")
                else:
                    print(f"  ⚠️  NO PREVIOUS FEATURES - First frame")
                    attended_feat = normalized_feat

                temporal_features.append(attended_feat)
                print(attended_feat.shape)

                # Add to memory for next frame
                self.temporal_memory.add(attended_feat)  # 1536 dims
            
            normalized_features = torch.cat(temporal_features, dim=0)
        else:
            # Original behavior when temporal attention is disabled
            concat_features = torch.cat((image_features, audio_features), dim=1)
            normalized_features = nn.functional.normalize(concat_features, dim=1)
        
        outputs = self.decoder.forward(normalized_features, captions)
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
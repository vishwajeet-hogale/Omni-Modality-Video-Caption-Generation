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
        self.use_text_feedback = getattr(cfg, 'use_text_feedback', True)  # Enable text feedback by default
        
        if self.use_temporal_attention:
            if self.use_text_feedback:
                # Enhanced temporal attention with text feedback
                self.text_encoder = BertModel.from_pretrained(cfg.text_encoder_cfg.encoder)
                self.text_projection = nn.Linear(cfg.text_encoder_cfg.input_dim, cfg.decoder_cfg.hidden_dim)
                self.text_tokenizer = BertTokenizer.from_pretrained(cfg.text_encoder_cfg.encoder)
                
                self.temporal_attention = EnhancedTemporalAttention(
                    multimodal_dim=cfg.decoder_cfg.hidden_dim,  # Projected dimension (768)
                    text_dim=cfg.decoder_cfg.hidden_dim,
                    hidden_dim=cfg.decoder_cfg.hidden_dim,
                    num_heads=getattr(cfg, 'temporal_attention_heads', 8),
                    dropout=getattr(cfg, 'temporal_attention_dropout', 0.1)
                )
                self.temporal_memory = EnhancedTemporalMemory(
                    max_len=getattr(cfg, 'temporal_memory_length', 10),
                    text_dim=cfg.decoder_cfg.hidden_dim
                )
            else:
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
        self.projection = nn.Linear(2304, 1536)  # Input: 2304, Output: 1536
    def forward(self, images, captions, audio, is_new_video=False, previous_captions=None):
        B, device = images.size(0), images.device

        # Reset temporal memory if starting a new video
        if is_new_video and self.use_temporal_attention:
            self.temporal_memory.clear()

        # 1) encode multimodal features
        img_feats = self.encoder(images)          # [B, I]
        aud_feats = self.audio_encoder(audio)     # [B, A]

        # 2) fuse + normalize
        current_mm_features = torch.cat([img_feats, aud_feats], dim=1)   # [B, 1536]
        current_mm_features = F.normalize(current_mm_features, dim=1)

        # 3) encode previous caption if available (for enhanced temporal attention)
        current_text_features = None
        if self.use_temporal_attention and self.use_text_feedback and previous_captions is not None:
            # Tokenize previous captions
            tokenized = self.text_tokenizer(
                previous_captions, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=60
            ).to(device)
            
            # Encode with BERT
            text_outputs = self.text_encoder(**tokenized)
            current_text_features = text_outputs.last_hidden_state.mean(dim=1)  # [B, 768]
            current_text_features = self.text_projection(current_text_features)  # [B, 768]

        # 4) apply temporal attention
        enhanced_features = current_mm_features
        if self.use_temporal_attention:
            if not hasattr(self, 'memory_projection'):
                self.memory_projection = nn.Linear(1536, 768).to(device)
            memory_features = self.memory_projection(current_mm_features)  # 1536 -> 768
            self.temporal_memory.add(memory_features)  # Store projected features
            if self.use_text_feedback:
                # Enhanced temporal attention with text feedback
                mm_memory, text_memory, mask = self.temporal_memory.get_previous(B, device)
                if mm_memory is not None:
                    # Ensure batch sizes match for both mm_memory and text_memory
                    if mm_memory.size(0) != B or (text_memory is not None and text_memory.size(0) != B):
                        # Use only the current features without temporal attention
                        enhanced_features = current_mm_features
                    else:
                        try:
                            # Project current features to match memory dimensions
                            current_mm_projected = self.memory_projection(current_mm_features)  # 1536 -> 768
                            current_text_projected = self.text_projection(current_text_features) if current_text_features is not None else None
                            
                            enhanced_features = self.temporal_attention(
                                current_mm_projected, 
                                current_text_projected,
                                mm_memory, 
                                text_memory, 
                                mask
                            )
                        except RuntimeError as e:
                            # If there's a tensor size mismatch, fall back to current features
                            print(f"Warning: Temporal attention failed due to tensor mismatch: {e}")
                            enhanced_features = current_mm_features
                else:
                    enhanced_features = current_mm_features
                
                # Only apply projection if we used temporal attention (768 -> 1536)
                # If we fell back to current_mm_features, it's already 1536 dimensions
                if enhanced_features.size(-1) == 768:  # Temporal attention was used
                    if not hasattr(self, 'enhanced_projection'):
                        self.enhanced_projection = nn.Linear(768, 1536).to(device)
                    enhanced_features = self.enhanced_projection(enhanced_features)
                # If enhanced_features is already 1536, no projection needed
            else:
                # Standard temporal attention without text feedback
                memory, mask = self.temporal_memory.get_previous(B, device)
                if memory is not None:
                    attn_results = self.temporal_attention(query=current_mm_features, memory=memory, mask=mask)
                    combined_prev_current = torch.concat([current_mm_features, attn_results], dim=1)
                    enhanced_features = self.projection(combined_prev_current)  # 2304 -> 1536

        # 5) decode this frame
        outputs = self.decoder.forward(enhanced_features, captions)

        # 6) update memory with current frame features (for next frame)
        if self.use_temporal_attention:
            if self.use_text_feedback:
                # Store both multimodal and text features
                with torch.no_grad():
                    # Store projected features (768 dims) to match what we stored earlier
                    projected_mm = self.memory_projection(current_mm_features.detach())  # 1536 -> 768
                    projected_text = self.text_projection(current_text_features.detach()) if current_text_features is not None else None
                    self.temporal_memory.add(
                        projected_mm,  # 768 dims
                        projected_text  # 768 dims
                    )
            else:
                # Store only multimodal features (original behavior)
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
            
        # Encode current frame
        image_features = self.encoder(images)
        audio_features = self.audio_encoder(audio)
        current_mm_features = torch.cat((image_features, audio_features), dim=1)
        current_mm_features = F.normalize(current_mm_features, dim=1)
        
        # Encode previous caption for text feedback
        current_text_features = None
        if self.use_temporal_attention and self.use_text_feedback and previous_caption is not None:
            tokenized = self.text_tokenizer(
                [previous_caption] * B,  # Repeat for batch
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=60
            ).to(device)
            
            text_outputs = self.text_encoder(**tokenized)
            current_text_features = text_outputs.last_hidden_state.mean(dim=1)
            current_text_features = self.text_projection(current_text_features)
        
        # Apply temporal attention
        enhanced_features = current_mm_features
        if self.use_temporal_attention:
            if self.use_text_feedback:
                # Enhanced temporal attention with text feedback
                mm_memory, text_memory, mask = self.temporal_memory.get_previous(B, device)
                if mm_memory is not None:
                    try:
                        # Project current features to match memory dimensions
                        current_mm_projected = self.memory_projection(current_mm_features)  # 1536 -> 768
                        current_text_projected = self.text_projection(current_text_features) if current_text_features is not None else None
                        
                        enhanced_features = self.temporal_attention(
                            current_mm_projected,
                            current_text_projected,
                            mm_memory,
                            text_memory,
                            mask
                        )
                    except RuntimeError as e:
                        # If there's a tensor size mismatch, fall back to current features
                        print(f"Warning: Temporal attention failed due to tensor mismatch: {e}")
                        enhanced_features = current_mm_features
            else:
                # Standard temporal attention
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
            if self.use_text_feedback:
                # Store both multimodal and text features
                with torch.no_grad():
                    self.temporal_memory.add(
                        current_mm_features.detach(),
                        current_text_features.detach() if current_text_features is not None else None
                    )
            else:
                # Store only multimodal features
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
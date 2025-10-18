import torch.nn as nn
import timm
import torch
from peft import get_peft_model, LoraConfig, TaskType

class VisionEncoder(nn.Module):
    def __init__(self, encoder_name: str, output_dim: int, projection_dim: int, use_lora: bool = False, lora_r: int = 8):
        super().__init__()
        self.vision_model = timm.create_model(encoder_name, pretrained=True)
        self.use_lora = use_lora
        
        if use_lora:
            print(f"🚀 Enabling LoRA for SwinTransformer with r={lora_r}")
            # Configure LoRA for SwinTransformer - use generic patterns
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=16,
                target_modules=["Linear"],  # Target all Linear layers
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            
            # Apply LoRA to the vision model
            try:
                self.vision_model = get_peft_model(self.vision_model, lora_config)
                print(f"✅ LoRA applied to SwinTransformer with {lora_r} rank")
            except Exception as e:
                print(f"❌ LoRA application failed: {e}")
                print(f"🔄 Falling back to frozen SwinTransformer")
                # Fallback to frozen model
                for param in self.vision_model.parameters():
                    param.requires_grad = False
                self.vision_model.eval()
            
            # Print LoRA model info for debugging
            trainable_params = sum(p.numel() for p in self.vision_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.vision_model.parameters())
            print(f"📊 LoRA Model: {trainable_params:,} trainable / {total_params:,} total parameters")
        else:
            # Freeze the vision model to save compute
            for param in self.vision_model.parameters():
                param.requires_grad = False
            self.vision_model.eval()
        
        self.projection = nn.Linear(output_dim, projection_dim)

    def forward(self, images):
        if self.use_lora:
            # LoRA model can have gradients, so no no_grad
            try:
                features = self.vision_model(images)
            except Exception as e:
                print(f"❌ LoRA forward pass failed: {e}")
                print(f"🔄 Trying with no_grad...")
                with torch.no_grad():
                    features = self.vision_model(images)
        else:
            # Use no_grad for frozen vision model to save memory and compute
            with torch.no_grad():
                features = self.vision_model(images)
        
        # Projection layer always has gradients
        projected = self.projection(features)
        return projected

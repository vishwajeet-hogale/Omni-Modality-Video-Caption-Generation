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
            # Temporarily disable LoRA for SwinTransformer due to compatibility issues
            print(f"⚠️  LoRA disabled for SwinTransformer due to compatibility issues")
            print(f"⚠️  Using frozen SwinTransformer instead")
            # Freeze the vision model to save compute
            for param in self.vision_model.parameters():
                param.requires_grad = False
            self.vision_model.eval()
        else:
            # Freeze the vision model to save compute
            for param in self.vision_model.parameters():
                param.requires_grad = False
            self.vision_model.eval()
        
        self.projection = nn.Linear(output_dim, projection_dim)

    def forward(self, images):
        # Use no_grad for frozen vision model to save memory and compute
        with torch.no_grad():
            features = self.vision_model(images)
        
        # Only the projection layer will have gradients
        projected = self.projection(features)
        return projected

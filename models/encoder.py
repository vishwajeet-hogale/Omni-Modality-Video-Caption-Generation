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
            # Configure LoRA for SwinTransformer
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=16,
                target_modules=["qkv", "proj", "mlp.fc1", "mlp.fc2"],  # SwinTransformer specific modules
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            
            # Apply LoRA to the vision model
            self.vision_model = get_peft_model(self.vision_model, lora_config)
            print(f"✅ LoRA applied to SwinTransformer with {lora_r} rank")
        else:
            # Freeze the vision model to save compute
            for param in self.vision_model.parameters():
                param.requires_grad = False
            self.vision_model.eval()
        
        self.projection = nn.Linear(output_dim, projection_dim)

    def forward(self, images):
        if self.use_lora:
            # LoRA model can have gradients, so no no_grad
            features = self.vision_model(images)
        else:
            # Use no_grad for frozen vision model to save memory and compute
            with torch.no_grad():
                features = self.vision_model(images)
        
        # Projection layer always has gradients
        projected = self.projection(features)
        return projected

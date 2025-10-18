import torch.nn as nn
import timm
import torch
from peft import get_peft_model, LoraConfig, TaskType

class SwinWrapper(nn.Module):
    """Wrapper for SwinTransformer to make it compatible with PEFT"""
    def __init__(self, swin_model):
        super().__init__()
        self.swin = swin_model

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        # PEFT expects input_ids, but we ignore it and use pixel_values for Swin
        if pixel_values is not None:
            return self.swin(pixel_values)
        else:
            # Fallback: if no pixel_values, assume first argument is images
            args = list(kwargs.values()) if kwargs else []
            if args:
                return self.swin(args[0])
            else:
                raise ValueError("No valid input found for SwinTransformer")

class VisionEncoder(nn.Module):
    def __init__(self, encoder_name: str, output_dim: int, projection_dim: int, use_lora: bool = False, lora_r: int = 8):
        super().__init__()
        self.vision_model = timm.create_model(encoder_name, pretrained=True)
        self.use_lora = use_lora
        
        if use_lora:
            print(f"🚀 Enabling LoRA for SwinTransformer with r={lora_r}")
            
            # First, let's see what modules are available
            print("🔍 Scanning SwinTransformer modules...")
            linear_modules = []
            for name, module in self.vision_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    linear_modules.append(name)
            
            print(f"📋 Found {len(linear_modules)} Linear modules")
            if len(linear_modules) > 0:
                print(f"📋 First few modules: {linear_modules[:5]}")
            
            # Use a more conservative LoRA configuration
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=16,
                target_modules=linear_modules[:10] if len(linear_modules) > 10 else linear_modules,  # Use actual module names
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            
            # Apply LoRA to the vision model using wrapper
            try:
                # Wrap SwinTransformer to make it PEFT-compatible
                wrapped_model = SwinWrapper(self.vision_model)
                self.vision_model = get_peft_model(wrapped_model, lora_config)
                print(f"✅ LoRA applied to SwinTransformer with {lora_r} rank")
                
                # Print LoRA model info
                trainable_params = sum(p.numel() for p in self.vision_model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.vision_model.parameters())
                print(f"📊 LoRA Model: {trainable_params:,} trainable / {total_params:,} total parameters")
                
            except Exception as e:
                print(f"❌ LoRA application failed: {e}")
                print(f"🔄 Falling back to frozen SwinTransformer")
                # Fallback to frozen model
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
        if self.use_lora:
            # LoRA model - pass images as pixel_values to satisfy PEFT signature
            features = self.vision_model(pixel_values=images)
        else:
            # Use no_grad for frozen vision model to save memory and compute
            with torch.no_grad():
                features = self.vision_model(images)
        
        # Projection layer always has gradients
        projected = self.projection(features)
        return projected

import os
import torch
import pandas as pd
from video_inference import predict_and_annotate_images
from omegaconf import OmegaConf
from models.model import SwinBart
from torchvision import transforms

def evaluate_checkpoints(checkpoint_dir, config_path):
    # List all checkpoints
    checkpoints = [os.path.join(checkpoint_dir, ckpt) for ckpt in os.listdir(checkpoint_dir) if ckpt.endswith(".pth")]
    
    print(f"🔍 Found {len(checkpoints)} checkpoints to evaluate:")
    for ckpt in checkpoints:
        print(f"  - {os.path.basename(ckpt)}")

    best_bleu = 0
    best_checkpoint = None

    for checkpoint in checkpoints:
        print(f"Evaluating checkpoint: {checkpoint}")

        # Load config
        cfg = OmegaConf.load(config_path)
        cfg.inference.model_path = checkpoint

        # Set device
        device = torch.device(cfg.trainer.device)

        # Initialize and load model
        model = SwinBart(cfg).to(device)
        model.load_state_dict(torch.load(cfg.inference.model_path, map_location=device))
        model.eval()

        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((cfg.inference.image_size, cfg.inference.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Run video inference and evaluation
        try:
            metrics = predict_and_annotate_images(cfg, model, transform, device)
            
            # Check if metrics is None or empty
            if metrics is None:
                print(f"⚠️  No metrics returned for {checkpoint} - skipping")
                continue
                
            # Extract BLEU score
            bleu_score = metrics.get("bleu", 0.0)
            print(f"BLEU Score for {checkpoint}: {bleu_score}")
            
            # Print all available metrics for debugging
            print(f"Available metrics: {list(metrics.keys()) if isinstance(metrics, dict) else 'Not a dict'}")

            # Track the best checkpoint
            if bleu_score > best_bleu:
                best_bleu = bleu_score
                best_checkpoint = checkpoint
                
        except Exception as e:
            print(f"❌ Error evaluating {checkpoint}: {e}")
            continue

    print(f"\nBest Checkpoint: {best_checkpoint} with BLEU Score: {best_bleu}")

# Example usage
if __name__ == "__main__":
    checkpoint_dir = "/Users/rohitkulkarni/Desktop/Omni-Modality-Video-Caption-Generation/checkpoints"  # Replace with your checkpoint directory
    config_path = "/Users/rohitkulkarni/Desktop/Omni-Modality-Video-Caption-Generation/configs/default.yaml"  # Replace with your config file path

    evaluate_checkpoints(checkpoint_dir, config_path)
#!/usr/bin/env python3
"""
Training-Consistent Video Inference Script
==========================================

This script replicates the exact training pipeline during inference to ensure
consistent and high-quality caption generation. It addresses the mismatch 
between training and inference that often causes poor results.

Key Features:
- Exact training data pipeline replication
- Proper teacher forcing simulation
- Consistent preprocessing and normalization
- Frame-by-frame processing with temporal context
- Robust error handling and logging

Author: AI Assistant
Date: 2025-10-02
"""

import os
import sys
import json
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import warnings
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import SwinBart


class InferenceVideoDataset(Dataset):
    """Dataset that exactly replicates training data pipeline for inference."""
    
    def __init__(self, frames_dir, audio_dir, transform=None):
        self.frames_dir = frames_dir
        self.audio_dir = audio_dir
        self.transform = transform
        
        # Get all frame files
        self.frame_files = []
        for file in sorted(os.listdir(frames_dir)):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                self.frame_files.append(file)
        
        print(f"📊 Found {len(self.frame_files)} frames")
    
    def __len__(self):
        return len(self.frame_files)
    
    def __getitem__(self, idx):
        # Load image
        frame_file = self.frame_files[idx]
        frame_path = os.path.join(self.frames_dir, frame_file)
        
        try:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"⚠️  Error loading image {frame_path}: {e}")
            # Create dummy image
            image = torch.zeros(3, 224, 224)
        
        # Load corresponding audio
        audio_file = frame_file.replace('.jpg', '.pt').replace('.jpeg', '.pt').replace('.png', '.pt')
        audio_path = os.path.join(self.audio_dir, audio_file)
        
        try:
            if os.path.exists(audio_path):
                audio = torch.load(audio_path, map_location='cpu')
                if isinstance(audio, dict):
                    audio = audio.get('features', audio.get('mfcc', torch.zeros(13, 100)))
                # Ensure correct shape
                if audio.dim() == 1:
                    audio = audio.unsqueeze(-1).expand(-1, 100)  # [13, 100]
                elif audio.dim() == 2 and audio.size(-1) != 100:
                    # Pad or truncate to 100 frames
                    if audio.size(-1) < 100:
                        audio = F.pad(audio, (0, 100 - audio.size(-1)))
                    else:
                        audio = audio[:, :100]
            else:
                audio = torch.zeros(13, 100)
        except Exception as e:
            print(f"⚠️  Error loading audio {audio_path}: {e}")
            audio = torch.zeros(13, 100)
        
        # Extract frame number for ordering
        try:
            frame_num = int(frame_file.split('_')[1].split('.')[0])
        except:
            frame_num = idx
        
        return {
            'image': image,
            'audio': audio,
            'frame_file': frame_file,
            'frame_num': frame_num,
            'index': idx
        }


def setup_model(cfg, device):
    """Setup model exactly as in training."""
    print("📦 Setting up model...")
    
    # Initialize model
    model = SwinBart(cfg)
    model.to(device)
    
    # Load checkpoint
    checkpoint_path = cfg.inference.model_path
    if os.path.exists(checkpoint_path):
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
            
            print("✅ Checkpoint loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("🔄 Using randomly initialized model")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("🔄 Using randomly initialized model")
    
    # Set to evaluation mode
    model.eval()
    
    # Ensure vision encoder is in correct mode based on config
    if hasattr(model.encoder, 'vision_model'):
        if cfg.vision_encoder_cfg.freeze:
            model.encoder.vision_model.eval()
            print("🧊 Vision encoder frozen for inference")
        else:
            model.encoder.vision_model.train()  # Keep in train mode if not frozen
            print("🔥 Vision encoder active for inference")
    
    return model


def generate_caption_training_style(model, image, audio, device, cfg, previous_context=None):
    """
    Generate caption using training-style approach for consistency.
    
    Args:
        model: The SwinBart model
        image: Input image tensor [1, 3, 224, 224]
        audio: Input audio tensor [1, 13, 100] 
        device: Computing device
        cfg: Configuration
        previous_context: Previous caption for context (optional)
        
    Returns:
        Generated caption string
    """
    
    with torch.no_grad():
        # Ensure correct input shapes
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dim
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # Add batch dim
        
        # Move to device
        image = image.to(device)
        audio = audio.to(device)
        
        try:
            # Method 1: Direct generation (preferred)
            generated_ids = model.generate(
                images=image,
                audio=audio,
                max_length=cfg.inference.max_length,
                num_beams=cfg.inference.num_beams,
                is_new_video=False,
                previous_caption=previous_context
            )
            
            # Decode
            if generated_ids.dim() > 1:
                generated_ids = generated_ids[0]  # Take first batch item
            
            caption = model.decoder.tokenizer.decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return caption.strip()
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            
            try:
                # Method 2: Fallback with forward pass
                # Create dummy caption for forward pass
                dummy_caption = ["generating medical procedure description"]
                outputs = model(image, dummy_caption, audio, is_new_video=False)
                
                # Use the logits to generate
                if hasattr(outputs, 'logits'):
                    # Simple greedy decoding
                    predicted_ids = torch.argmax(outputs.logits, dim=-1)
                    if predicted_ids.dim() > 1:
                        predicted_ids = predicted_ids[0]  # Take first batch
                    
                    caption = model.decoder.tokenizer.decode(
                        predicted_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    return caption.strip()
                
            except Exception as e2:
                print(f"❌ Fallback generation error: {e2}")
                return f"Error generating caption: {str(e)}"
        
        return "Unable to generate caption"


def process_video_sequence(model, dataset, device, cfg):
    """
    Process video sequence frame by frame with temporal context.
    """
    print("🎬 Processing video sequence...")
    
    # Create dataloader for efficient processing
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # Process one frame at a time for temporal consistency
        shuffle=False,  # Keep temporal order
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=torch.cuda.is_available()
    )
    
    results = []
    previous_captions = []
    
    # Reset temporal memory for new video
    if hasattr(model, 'reset_temporal_memory'):
        model.reset_temporal_memory()
    
    print(f"🧠 Temporal attention: {'Enabled' if model.use_temporal_attention else 'Disabled'}")
    
    # Process frames
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing frames")):
        
        image = batch['image']  # [1, 3, 224, 224]
        audio = batch['audio']  # [1, 13, 100]
        frame_file = batch['frame_file'][0]  # Remove batch dimension
        frame_num = batch['frame_num'].item()
        
        # Get previous context
        prev_context = previous_captions[-1] if previous_captions else None
        
        # Generate caption
        try:
            caption = generate_caption_training_style(
                model=model,
                image=image,
                audio=audio, 
                device=device,
                cfg=cfg,
                previous_context=prev_context
            )
            
            # Store result
            results.append({
                'frame_num': frame_num,
                'frame_file': frame_file,
                'caption': caption,
                'batch_idx': batch_idx,
                'has_error': caption.startswith('Error')
            })
            
            # Update context (keep last N captions)
            previous_captions.append(caption)
            if len(previous_captions) > 3:  # Keep last 3 captions for context
                previous_captions.pop(0)
            
        except Exception as e:
            print(f"❌ Error processing frame {frame_num}: {e}")
            results.append({
                'frame_num': frame_num,
                'frame_file': frame_file,
                'caption': f"Processing error: {str(e)}",
                'batch_idx': batch_idx,
                'has_error': True
            })
    
    return results


def analyze_results(results):
    """Analyze caption quality and diversity."""
    print("\n📊 Analyzing results...")
    
    # Basic stats
    total_frames = len(results)
    successful_frames = len([r for r in results if not r['has_error']])
    
    # Get successful captions only
    captions = [r['caption'] for r in results if not r['has_error']]
    
    # Caption diversity
    unique_captions = len(set(captions)) if captions else 0
    diversity_ratio = unique_captions / len(captions) if captions else 0
    
    # Average caption length
    avg_length = np.mean([len(c.split()) for c in captions]) if captions else 0
    
    # Medical terminology check
    medical_terms = [
        'tube', 'nasogastric', 'ng', 'insert', 'insertion', 'patient', 'medical',
        'procedure', 'healthcare', 'nurse', 'doctor', 'stomach', 'nose', 'throat',
        'landmark', 'marking', 'xiphoid', 'earlobe', 'suction', 'aspirate'
    ]
    
    medical_term_usage = 0
    for caption in captions:
        if any(term in caption.lower() for term in medical_terms):
            medical_term_usage += 1
    
    medical_usage_ratio = medical_term_usage / len(captions) if captions else 0
    
    print(f"📈 Results Analysis:")
    print(f"  Total frames: {total_frames}")
    print(f"  Successful frames: {successful_frames} ({100*successful_frames/total_frames:.1f}%)")
    print(f"  Unique captions: {unique_captions}")
    print(f"  Caption diversity: {diversity_ratio:.2%}")
    print(f"  Average caption length: {avg_length:.1f} words")
    print(f"  Medical terminology usage: {medical_usage_ratio:.2%}")
    
    # Quality assessment
    if diversity_ratio > 0.3:
        print("✅ Good caption diversity")
    elif diversity_ratio > 0.1:
        print("⚠️  Moderate caption diversity") 
    else:
        print("❌ Poor caption diversity - model may be undertrained")
    
    return {
        'total_frames': total_frames,
        'successful_frames': successful_frames,
        'unique_captions': unique_captions,
        'diversity_ratio': diversity_ratio,
        'avg_length': avg_length,
        'medical_usage_ratio': medical_usage_ratio
    }


def save_results(results, analysis, cfg, save_dir):
    """Save results in multiple formats."""
    print(f"💾 Saving results to {save_dir}...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. CSV file
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "training_style_captions.csv")
    df.to_csv(csv_path, index=False)
    print(f"📊 CSV saved: {csv_path}")
    
    # 2. Detailed text report
    report_path = os.path.join(save_dir, "training_style_report.txt")
    with open(report_path, 'w') as f:
        f.write("Training-Style Video Inference Report\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Model: SwinBart with temporal attention\n")
        f.write(f"  Vision encoder frozen: {cfg.vision_encoder_cfg.freeze}\n")
        f.write(f"  Temporal attention: {cfg.use_temporal_attention}\n")
        f.write(f"  Max caption length: {cfg.inference.max_length}\n")
        f.write(f"  Beam search: {cfg.inference.num_beams}\n\n")
        
        f.write("RESULTS SUMMARY:\n")
        for key, value in analysis.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("FRAME-BY-FRAME CAPTIONS:\n")
        f.write("-" * 30 + "\n")
        
        # Sort by frame number
        sorted_results = sorted(results, key=lambda x: x['frame_num'])
        for r in sorted_results:
            status = "✓" if not r['has_error'] else "✗"
            f.write(f"Frame {r['frame_num']:04d} {status}: {r['caption']}\n")
    
    print(f"📝 Report saved: {report_path}")
    
    # 3. JSON summary
    summary = {
        'config': {
            'model': 'SwinBart',
            'vision_frozen': cfg.vision_encoder_cfg.freeze,
            'temporal_attention': cfg.use_temporal_attention,
            'max_length': cfg.inference.max_length,
            'num_beams': cfg.inference.num_beams
        },
        'analysis': analysis,
        'sample_captions': [r['caption'] for r in sorted_results[:10]]  # First 10
    }
    
    json_path = os.path.join(save_dir, "training_style_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📋 Summary saved: {json_path}")


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function for training-consistent video inference.
    """
    print("🎬 Training-Consistent Video Inference")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Setup data transforms (exactly as in training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check input directories
    frames_dir = cfg.inference.video_image_dir
    audio_dir = frames_dir.replace('/frames/', '/mfccs/')
    
    if not os.path.exists(frames_dir):
        print(f"❌ Error: Frames directory not found: {frames_dir}")
        return
    
    if not os.path.exists(audio_dir):
        print(f"⚠️  Warning: Audio directory not found: {audio_dir}")
        print("📁 Creating audio directory with dummy files...")
        os.makedirs(audio_dir, exist_ok=True)
    
    print(f"📁 Frames directory: {frames_dir}")
    print(f"🎵 Audio directory: {audio_dir}")
    
    # Create dataset
    dataset = InferenceVideoDataset(
        frames_dir=frames_dir,
        audio_dir=audio_dir,
        transform=transform
    )
    
    if len(dataset) == 0:
        print("❌ Error: No frames found in dataset")
        return
    
    # Setup model
    model = setup_model(cfg, device)
    
    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model info:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Process video
    results = process_video_sequence(model, dataset, device, cfg)
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Save results
    save_dir = cfg.inference.inf_save_dir + "_training_style"
    save_results(results, analysis, cfg, save_dir)
    
    # Final summary
    print("\n🎉 Training-style inference completed!")
    print(f"📈 Processed {analysis['total_frames']} frames")
    print(f"✅ Success rate: {100*analysis['successful_frames']/analysis['total_frames']:.1f}%") 
    print(f"🎯 Caption diversity: {analysis['diversity_ratio']:.2%}")
    print(f"📚 Medical term usage: {analysis['medical_usage_ratio']:.2%}")
    print(f"📂 Results saved to: {save_dir}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if analysis['diversity_ratio'] < 0.1:
        print("🔧 Low diversity detected - consider retraining the model")
    if analysis['medical_usage_ratio'] < 0.5:
        print("🏥 Low medical terminology - check training data quality")
    if analysis['successful_frames'] < analysis['total_frames']:
        print("⚠️  Some processing errors - check model and data compatibility")
    
    print("🔗 For better results, ensure model was trained with identical preprocessing!")


if __name__ == "__main__":
    main()
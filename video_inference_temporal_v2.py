#!/usr/bin/env python3
"""
Video Inference with Temporal Attention - Version 2
===================================================

This script performs video captioning with temporal attention mechanism.
It processes video frames sequentially, maintaining temporal context across frames.

Features:
- Robust device detection (MPS > CUDA > CPU)
- Proper device management for all tensors
- Enhanced error handling and logging
- Temporal memory management
- Video boundary detection
- Comprehensive progress tracking

Author: AI Assistant
Date: 2025-10-02
"""

import os
import sys
import json
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import SwinBart
from models.temporal_attention import EnhancedTemporalMemory


def detect_video_boundary(current_frame_idx, previous_frame_idx, threshold=10):
    """
    Detect if we've moved to a new video sequence based on frame index gap.
    
    Args:
        current_frame_idx (int): Current frame index
        previous_frame_idx (int): Previous frame index
        threshold (int): Gap threshold to consider as new video
        
    Returns:
        bool: True if new video detected, False otherwise
    """
    if previous_frame_idx is None:
        return True  # First frame
    return (current_frame_idx - previous_frame_idx) > threshold


def get_device():
    """
    Get the best available device for computation.
    Priority: MPS > CUDA > CPU
    
    Returns:
        torch.device: The selected device
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using MPS (Metal Performance Shaders) for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU for computation")
    
    return device


def load_model(cfg, device):
    """
    Load the SwinBart model with proper device management.
    
    Args:
        cfg: Configuration object
        device: Target device for the model
        
    Returns:
        SwinBart: Loaded model
    """
    print("📦 Loading SwinBart model...")
    
    # Initialize model
    model = SwinBart(cfg)
    model.to(device)
    model.eval()
    
    # Load checkpoint if available
    if os.path.exists(cfg.inference.model_path):
        print(f"📂 Loading checkpoint from: {cfg.inference.model_path}")
        try:
            checkpoint = torch.load(cfg.inference.model_path, map_location=device)
            
            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
                
            # Remove unexpected keys if they exist
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            
            # Load the filtered state dict
            model.load_state_dict(state_dict, strict=False)
            print("✅ Model checkpoint loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load checkpoint: {e}")
            print("🔄 Using randomly initialized model")
    else:
        print(f"⚠️  Warning: Checkpoint not found at {cfg.inference.model_path}")
        print("🔄 Using randomly initialized model")
    
    return model


def load_image_batch(image_paths, transform, device):
    """
    Load and preprocess a batch of images.
    
    Args:
        image_paths (list): List of image file paths
        transform: Image transformation pipeline
        device: Target device
        
    Returns:
        torch.Tensor: Batch of preprocessed images
    """
    images = []
    for path in image_paths:
        try:
            image = Image.open(path).convert('RGB')
            image = transform(image)
            images.append(image)
        except Exception as e:
            print(f"⚠️  Warning: Could not load image {path}: {e}")
            # Create a dummy image if loading fails
            dummy_image = torch.zeros(3, 224, 224)
            images.append(dummy_image)
    
    return torch.stack(images).to(device)


def load_audio_batch(audio_paths, device):
    """
    Load and preprocess a batch of audio features.
    
    Args:
        audio_paths (list): List of audio file paths
        device: Target device
        
    Returns:
        torch.Tensor: Batch of audio features
    """
    audios = []
    for path in audio_paths:
        try:
            if os.path.exists(path):
                audio = torch.load(path, map_location=device)
                if isinstance(audio, dict):
                    audio = audio.get('features', audio.get('mfcc', torch.zeros(13, 100)))
                audios.append(audio)
            else:
                # Create dummy audio if file doesn't exist
                dummy_audio = torch.zeros(13, 100)
                audios.append(dummy_audio)
        except Exception as e:
            print(f"⚠️  Warning: Could not load audio {path}: {e}")
            # Create dummy audio if loading fails
            dummy_audio = torch.zeros(13, 100)
            audios.append(dummy_audio)
    
    return torch.stack(audios).to(device)


def extract_frame_info(image_path):
    """
    Extract frame index from image path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        int: Frame index
    """
    try:
        filename = os.path.basename(image_path)
        # Extract frame number from filename like "frame_0064.jpg"
        frame_num = int(filename.split('_')[1].split('.')[0])
        return frame_num
    except:
        return 0


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function for video inference with temporal attention.
    
    Args:
        cfg: Hydra configuration object
    """
    print("🎬 Video Inference with Temporal Attention - Version 2")
    print("=" * 60)
    
    # Get device
    device = get_device()
    
    # Load model
    model = load_model(cfg, device)
    
    # Image preprocessing
    print("🖼️  Setting up image preprocessing...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get video directory
    video_dir = cfg.inference.video_image_dir
    if not os.path.exists(video_dir):
        print(f"❌ Error: Video directory not found: {video_dir}")
        return
    
    print(f"📁 Processing video directory: {video_dir}")
    
    # Get all image paths
    image_paths = []
    for root, dirs, files in os.walk(video_dir):
        for file in sorted(files):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print("❌ Error: No images found in video directory")
        return
    
    print(f"📊 Found {len(image_paths)} frames to process")
    
    # Get corresponding audio paths
    audio_paths = []
    for img_path in image_paths:
        # Convert image path to audio path
        audio_path = img_path.replace('/frames/', '/mfccs/').replace('.jpg', '.pt').replace('.jpeg', '.pt').replace('.png', '.pt')
        audio_paths.append(audio_path)
    
    # Create output directory
    save_dir = cfg.inference.inf_save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 Output directory: {save_dir}")
    
    # Initialize results storage
    results = []
    previous_captions = []
    previous_frame_idx = None
    
    # Process frames in batches
    batch_size = cfg.trainer.batch_size
    print(f"🔄 Processing in batches of {batch_size}")
    print(f"🧠 Temporal attention: {'Enabled' if model.use_temporal_attention else 'Disabled'}")
    print(f"📝 Text feedback: {'Enabled' if model.use_text_feedback else 'Disabled'}")
    
    # Progress bar
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    progress_bar = tqdm(range(0, len(image_paths), batch_size), desc="Processing batches", unit="batch")
    
    for batch_start in progress_bar:
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_paths = image_paths[batch_start:batch_end]
        batch_audio_paths = audio_paths[batch_start:batch_end]
        
        # Load batch data
        try:
            images_batch = load_image_batch(batch_paths, transform, device)
            audios_batch = load_audio_batch(batch_audio_paths, device)
        except Exception as e:
            print(f"❌ Error loading batch {batch_start//batch_size + 1}: {e}")
            continue
        
        # Detect video boundary
        current_frame_idx = extract_frame_info(batch_paths[0])
        is_new_video = detect_video_boundary(current_frame_idx, previous_frame_idx)
        
        if is_new_video:
            model.reset_temporal_memory()
            print(f"🔄 New video sequence detected at frame {current_frame_idx}")
            previous_captions = []  # Reset captions for new video
        
        # Process each frame in the batch
        with torch.no_grad():
            for i in range(len(batch_paths)):
                current_image = images_batch[i:i+1]
                current_audio = audios_batch[i:i+1]
                
                # Get previous caption for context
                prev_cap = previous_captions[-1] if previous_captions else None
                
                try:
                    # Generate caption
                    generated_ids = model.generate(
                        current_image,
                        current_audio,
                        max_length=cfg.inference.max_length,
                        num_beams=cfg.inference.num_beams,
                        is_new_video=is_new_video,
                        previous_caption=prev_cap
                    )
                    
                    # Decode caption
                    caption = model.decoder.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    caption = caption.strip()
                    
                    # Store results
                    frame_idx = extract_frame_info(batch_paths[i])
                    results.append({
                        'frame_idx': frame_idx,
                        'image_path': batch_paths[i],
                        'caption': caption,
                        'is_new_video': is_new_video
                    })
                    
                    # Update previous captions
                    previous_captions.append(caption)
                    if len(previous_captions) > 5:  # Keep only last 5 captions
                        previous_captions.pop(0)
                    
                except Exception as e:
                    print(f"❌ Error generating caption for frame {i}: {e}")
                    # Add empty result to maintain consistency
                    frame_idx = extract_frame_info(batch_paths[i])
                    results.append({
                        'frame_idx': frame_idx,
                        'image_path': batch_paths[i],
                        'caption': "",
                        'is_new_video': is_new_video
                    })
                    previous_captions.append("")
        
        # Update progress
        progress_bar.set_postfix({
            'Frames': f"{batch_end}/{len(image_paths)}",
            'Results': len(results)
        })
        
        previous_frame_idx = current_frame_idx
    
    # Save results
    print("\n💾 Saving results...")
    
    # Save CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "temporal_captions_v2.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"📄 CSV saved: {csv_path}")
    
    # Save detailed report
    report_path = os.path.join(save_dir, "temporal_report_v2.txt")
    with open(report_path, 'w') as f:
        f.write("Video Captioning with Temporal Attention - Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total frames processed: {len(results)}\n")
        f.write(f"Video directory: {video_dir}\n")
        f.write(f"Model: SwinBart with Temporal Attention\n")
        f.write(f"Device: {device}\n\n")
        
        # Analyze temporal connectives
        temporal_connectives = ['then', 'next', 'after', 'before', 'while', 'during', 'meanwhile', 'subsequently', 'finally', 'initially']
        connective_count = 0
        
        f.write("Frame-by-frame captions:\n")
        f.write("-" * 30 + "\n")
        
        for r in results:
            if r['caption']:
                frame_connectives = [word for word in temporal_connectives if word in r['caption'].lower()]
                if frame_connectives:
                    connective_count += 1
                    f.write(f"Frame {r['frame_idx']:04d}: {', '.join(frame_connectives)} -> {r['caption']}\n")
                else:
                    f.write(f"Frame {r['frame_idx']:04d}: {r['caption']}\n")
        
        f.write(f"\nSummary:\n")
        f.write(f"Total frames: {len(results)}\n")
        f.write(f"Frames with temporal connectives: {connective_count}\n")
        if len(results) > 0:
            f.write(f"Narrative coherence: {connective_count/len(results)*100:.1f}%\n")
        else:
            f.write(f"Narrative coherence: 0.0% (no results)\n")
    
    print(f"📊 Report saved: {report_path}")
    
    # Print summary
    print("\n🎉 Processing completed!")
    print(f"📈 Total frames processed: {len(results)}")
    print(f"📄 Results saved to: {save_dir}")
    print(f"🔗 CSV file: {csv_path}")
    print(f"📊 Report file: {report_path}")


if __name__ == "__main__":
    main()

import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from models.model import SwinBart
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import re
import glob
from pathlib import Path

def load_mfcc(audio_path):
    """Load pre-computed MFCC features from .npy file"""
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found: {audio_path}")
        # Return dummy MFCC if file not found
        return torch.zeros(13, 100)  # Default shape: (features, time_steps)
    
    mfcc = np.load(audio_path)
    return torch.tensor(mfcc, dtype=torch.float32)

def extract_frame_number(filename):
    """Extract frame number from filename (e.g., frame_001.jpg -> 1)"""
    # Try different patterns
    patterns = [
        r'frame[_-]?(\d+)',
        r'(\d+)',
    ]
    
    basename = os.path.splitext(os.path.basename(filename))[0]
    
    for pattern in patterns:
        match = re.search(pattern, basename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return 0  # fallback

def get_frame_paths(frames_dir):
    """Get all frame paths and sort them by frame number"""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    frame_paths = []
    
    for ext in valid_extensions:
        frame_paths.extend(glob.glob(os.path.join(frames_dir, f"*{ext}")))
        frame_paths.extend(glob.glob(os.path.join(frames_dir, f"*{ext.upper()}")))
    
    # Sort by frame number
    frame_paths.sort(key=extract_frame_number)
    
    return frame_paths

def get_corresponding_audio_path(frame_path, frames_dir, audio_base_dir):
    """Get corresponding audio MFCC file for a frame"""
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]
    
    # Get the video folder name from frames_dir
    video_folder_name = os.path.basename(frames_dir)
    
    # Construct audio path: audio_base_dir/video_folder_name/frame_name.npy
    audio_path = os.path.join(audio_base_dir, video_folder_name, f"{frame_name}.npy")
    
    return audio_path

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = SwinBart(cfg)
    checkpoint = torch.load(cfg.inference.model_path, map_location=device)
    
    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load the state dict
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((cfg.inference.image_size, cfg.inference.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get video frames directory from config
    frames_dir = cfg.inference.video_image_dir
    audio_base_dir = cfg.data.audio_dir  # Base audio directory
    
    if not os.path.exists(frames_dir):
        raise ValueError(f"Frames directory not found: {frames_dir}")
    
    frame_paths = get_frame_paths(frames_dir)
    
    if not frame_paths:
        raise ValueError(f"No image files found in {frames_dir}")
    
    print(f"Found {len(frame_paths)} frames in {frames_dir}")
    print(f"Audio base directory: {audio_base_dir}")
    
    # Prepare output directory
    save_dir = cfg.inference.inf_save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Process frames with temporal context
    results = []
    
    print(f"Processing frames with temporal attention enabled: {cfg.use_temporal_attention}")
    print(f"Temporal memory length: {cfg.temporal_memory_length}")
    
    # Initialize temporal memory
    if hasattr(model, 'reset_temporal_memory'):
        model.reset_temporal_memory()
    
    previous_captions = []
    
    # Process frames sequentially
    for i in tqdm(range(len(frame_paths)), desc="Processing frames"):
        frame_path = frame_paths[i]
        frame_name = os.path.basename(frame_path)
        frame_number = extract_frame_number(frame_path)
        
        try:
            # Load and preprocess image
            image = Image.open(frame_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
            
            # Load corresponding audio features
            audio_path = get_corresponding_audio_path(frame_path, frames_dir, audio_base_dir)
            audio_tensor = load_mfcc(audio_path).unsqueeze(0).to(device)  # Add batch dimension
            
            # Prepare previous caption context (if text feedback is enabled)
            prev_caption = previous_captions[-1] if previous_captions and cfg.use_text_feedback else None
            
            # Generate caption
            with torch.no_grad():
                generated_ids = model.generate(
                    image_tensor,
                    audio_tensor,
                    max_length=cfg.inference.max_length,
                    num_beams=cfg.inference.num_beams,
                    previous_caption=prev_caption if cfg.use_text_feedback else None
                )
                
                # Decode caption
                caption = model.decoder.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Store result
            result = {
                'frame_number': frame_number,
                'frame_name': frame_name,
                'frame_path': frame_path,
                'audio_path': audio_path,
                'caption': caption,
                'previous_caption': prev_caption,
                'audio_exists': os.path.exists(audio_path)
            }
            results.append(result)
            
            # Update previous captions for text feedback
            if cfg.use_text_feedback:
                previous_captions.append(caption)
                # Keep only recent captions for context
                if len(previous_captions) > cfg.temporal_memory_length:
                    previous_captions.pop(0)
            
            # Show progress
            status = "✓" if os.path.exists(audio_path) else "⚠"
            print(f"Frame {i+1:04d}/{len(frame_paths):04d} {status} {frame_name}: {caption}")
            
        except Exception as e:
            print(f"❌ Error processing frame {frame_name}: {e}")
            continue
    
    # Save results
    print(f"\nSaving results to {save_dir}...")
    
    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "temporal_captions.csv")
    df.to_csv(csv_path, index=False)
    
    # Save as readable text
    txt_path = os.path.join(save_dir, "temporal_captions.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Video Commentary with Temporal Context\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Video: {os.path.basename(frames_dir)}\n")
        f.write(f"Total frames: {len(results)}\n")
        f.write(f"Temporal attention: {cfg.use_temporal_attention}\n")
        f.write(f"Text feedback: {cfg.use_text_feedback}\n")
        f.write(f"Temporal memory length: {cfg.temporal_memory_length}\n\n")
        
        for i, result in enumerate(results):
            audio_status = "✓" if result['audio_exists'] else "⚠ Missing"
            f.write(f"Frame {result['frame_number']:04d} ({result['frame_name']}) [Audio: {audio_status}]:\n")
            f.write(f"  Caption: {result['caption']}\n")
            if result['previous_caption'] and cfg.use_text_feedback:
                f.write(f"  Previous: {result['previous_caption']}\n")
            f.write(f"  Audio: {result['audio_path']}\n")
            f.write("\n")
    
    # Save narrative summary
    summary_path = os.path.join(save_dir, "narrative_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Narrative Summary\n")
        f.write("=" * 20 + "\n\n")
        
        # Extract key narrative elements
        all_captions = [r['caption'] for r in results]
        audio_found = sum(1 for r in results if r['audio_exists'])
        
        f.write("Configuration:\n")
        f.write(f"- Frames directory: {frames_dir}\n")
        f.write(f"- Audio directory: {audio_base_dir}\n")
        f.write(f"- Temporal attention: {cfg.use_temporal_attention}\n")
        f.write(f"- Text feedback: {cfg.use_text_feedback}\n")
        f.write(f"- Memory length: {cfg.temporal_memory_length}\n\n")
        
        f.write("Processing Summary:\n")
        f.write(f"- Total frames processed: {len(results)}\n")
        f.write(f"- Audio files found: {audio_found}/{len(results)}\n")
        f.write(f"- Success rate: {len(results)/len(frame_paths)*100:.1f}%\n")
        f.write(f"- Average caption length: {np.mean([len(c.split()) for c in all_captions]):.1f} words\n\n")
        
        f.write("Full Video Narrative:\n")
        f.write("-" * 20 + "\n")
        for i, caption in enumerate(all_captions):
            f.write(f"{i+1:3d}. {caption}\n")
    
    print(f"\nResults saved:")
    print(f"  📊 CSV: {csv_path}")
    print(f"  📝 Text: {txt_path}")
    print(f"  📖 Summary: {summary_path}")
    
    # Show statistics
    audio_found = sum(1 for r in results if r['audio_exists'])
    print(f"\n📈 Processing Statistics:")
    print(f"  ✅ Frames processed: {len(results)}/{len(frame_paths)}")
    print(f"  🎵 Audio files found: {audio_found}/{len(results)}")
    print(f"  🧠 Temporal attention: {'Enabled' if cfg.use_temporal_attention else 'Disabled'}")
    print(f"  💬 Text feedback: {'Enabled' if cfg.use_text_feedback else 'Disabled'}")
    
    # Show sample output
    if results:
        print(f"\n🎬 Sample captions:")
        print("-" * 30)
        for i in range(min(3, len(results))):
            r = results[i]
            status = "✓" if r['audio_exists'] else "⚠"
            print(f"Frame {r['frame_number']:04d} {status}: {r['caption']}")

if __name__ == "__main__":
    main()
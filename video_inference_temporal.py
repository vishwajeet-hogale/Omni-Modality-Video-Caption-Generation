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

def load_mfcc(audio_path):
    """Load pre-computed MFCC features from .npy file"""
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found: {audio_path}")
        # Return dummy MFCC if file not found
        return torch.zeros(1, 13, 100)  # Default shape
    
    mfcc = np.load(audio_path)
    if len(mfcc.shape) == 2:
        mfcc = mfcc[np.newaxis, ...]  # Add batch dimension
    return torch.tensor(mfcc, dtype=torch.float32)

def _parse_frame_idx(filename: str):
    """Parse frame index from filename like 'frame_0007.jpg' -> 7"""
    base = os.path.basename(filename)
    m = re.search(r'frame[_-]?(\d+)', base, flags=re.IGNORECASE)
    if not m:
        # fallback: last digit-run in the name
        m2 = re.findall(r'(\d+)', base)
        if not m2:
            return None
        return int(m2[-1])
    return int(m.group(1))

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = SwinBart(cfg)
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
    model.to(device)
    model.eval()
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((cfg.inference.image_size, cfg.inference.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get video frames directory
    video_dir = cfg.inference.video_image_dir
    valid_exts = {'.jpg', '.jpeg', '.png'}
    
    # Get all image paths and sort them by frame index for sequential processing
    image_paths = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if os.path.splitext(file)[-1].lower() in valid_exts:
                image_paths.append(os.path.join(root, file))
    
    # Sort by frame index to maintain temporal sequence
    image_paths.sort(key=lambda x: _parse_frame_idx(os.path.basename(x)) or 0)
    
    print(f"Processing {len(image_paths)} frames sequentially with temporal attention...")
    print(f"Text feedback enabled: {getattr(cfg, 'use_text_feedback', False)}")
    
    # Reset temporal memory for new video sequence
    model.reset_temporal_memory()
    
    results = []
    save_dir = os.path.join(cfg.inference.inf_save_dir, "temporal_captions")
    os.makedirs(save_dir, exist_ok=True)
    
    # Process frames in batches like training (batch_size from config)
    batch_size = cfg.data.batch_size
    previous_captions = []  # Store previous captions for the batch
    
    print(f"Processing in batches of {batch_size} to match training...")
    
    for batch_start in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_paths = image_paths[batch_start:batch_end]
        
        # Load batch data
        batch_images = []
        batch_audios = []
        batch_info = []
        
        for image_path in batch_paths:
            try:
                # Load and preprocess image
                image = Image.open(image_path).convert("RGB")
                image_tensor = transform(image)
                batch_images.append(image_tensor)
                
                # Load corresponding MFCC
                audio_path = image_path.replace(f"{os.sep}frames{os.sep}", f"{os.sep}mfccs{os.sep}")
                audio_path = os.path.splitext(audio_path)[0] + ".npy"
                audio_tensor = load_mfcc(audio_path)
                batch_audios.append(audio_tensor.squeeze(0))  # Remove batch dim for stacking
                
                frame_name = os.path.basename(image_path)
                frame_idx = _parse_frame_idx(frame_name)
                batch_info.append({
                    'frame_idx': frame_idx,
                    'frame_name': frame_name,
                    'image_path': image_path
                })
                
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue
        
        if not batch_images:
            continue
            
        # Stack into batch tensors (like training)
        images_batch = torch.stack(batch_images).to(device)  # [B, C, H, W]
        audios_batch = torch.stack(batch_audios).to(device)  # [B, n_mfcc, T]
        
        # Process batch with temporal context (exactly like training)
        with torch.no_grad():
            # Determine if this is a new video (reset at start of first batch)
            is_new_video = (batch_start == 0)
            
            # Reset temporal memory for new video
            if is_new_video:
                model.reset_temporal_memory()
                
            # Prepare previous captions for the batch (like training does)
            batch_previous_captions = None
            if len(previous_captions) > 0:
                # Use the last few captions as context (matching training)
                batch_previous_captions = previous_captions[-len(batch_images):]
                # Pad if necessary
                while len(batch_previous_captions) < len(batch_images):
                    batch_previous_captions.insert(0, None)
            
            # Create dummy captions for forward pass (we'll replace with generated ones)
            dummy_captions = [""] * len(batch_images)
            
            # Run forward pass to get features and update temporal memory
            # This matches the training process exactly
            outputs = model.forward(
                images_batch,
                dummy_captions,  # Will be ignored in generate mode
                audios_batch,
                is_new_video=is_new_video,
                previous_captions=batch_previous_captions
            )
            
            # Generate captions for each frame in the batch
            for i in range(len(batch_info)):
                # Get single frame
                current_image = images_batch[i:i+1]  # [1, C, H, W]
                current_audio = audios_batch[i:i+1]  # [1, n_mfcc, T]
                
                # Use previous caption if available
                prev_cap = batch_previous_captions[i] if batch_previous_captions and batch_previous_captions[i] else None
                
                # Generate caption
                generated_ids = model.generate(
                    images=current_image,
                    audio=current_audio,
                    max_length=cfg.inference.max_length,
                    num_beams=cfg.inference.num_beams,
                    previous_caption=prev_cap
                )
                
                # Decode caption
                caption = model.decoder.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Store results
                result_info = batch_info[i].copy()
                result_info['caption'] = caption
                result_info['previous_caption'] = prev_cap
                results.append(result_info)
                
                # Add to previous captions list
                previous_captions.append(caption)
                
                # Keep only last temporal_memory_length captions
                if len(previous_captions) > cfg.temporal_memory_length:
                    previous_captions.pop(0)
                
                frame_num = batch_start + i + 1
                print(f"Frame {frame_num}/{len(image_paths)} ({result_info['frame_name']}): {caption}")
                if prev_cap:
                    print(f"  Previous: {prev_cap}")
                print(f"  Batch {batch_start//batch_size + 1}, Frame {i+1}/{len(batch_info)}")
                print("-" * 60)
    
    # Save results
    # Save as CSV for analysis
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "temporal_captions.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Save as text file with narrative flow
    txt_path = os.path.join(save_dir, "temporal_captions.txt")
    with open(txt_path, 'w') as f:
        f.write("Video Commentary with Temporal Attention and Text Feedback\n")
        f.write("=" * 60 + "\n\n")
        
        for i, r in enumerate(results):
            f.write(f"Frame {r['frame_idx']:04d} ({r['frame_name']}):\n")
            f.write(f"Caption: {r['caption']}\n")
            if i > 0 and r.get('previous_caption'):
                f.write(f"Previous: {results[i-1]['caption']}\n")
            f.write("-" * 40 + "\n")
    
    # Save narrative flow analysis
    narrative_path = os.path.join(save_dir, "narrative_analysis.txt")
    with open(narrative_path, 'w') as f:
        f.write("Narrative Flow Analysis\n")
        f.write("=" * 30 + "\n\n")
        
        # Analyze temporal connectives
        temporal_words = ['first', 'next', 'then', 'after', 'finally', 'now', 'once', 'while', 'during']
        connective_count = 0
        
        for r in results:
            caption_lower = r['caption'].lower()
            frame_connectives = [word for word in temporal_words if word in caption_lower]
            if frame_connectives:
                connective_count += 1
                f.write(f"Frame {r['frame_idx']:04d}: {', '.join(frame_connectives)} -> {r['caption']}\n")
        
        f.write(f"\nSummary:\n")
        f.write(f"Total frames: {len(results)}\n")
        f.write(f"Frames with temporal connectives: {connective_count}\n")
        f.write(f"Narrative coherence: {connective_count/len(results)*100:.1f}%\n")
    
    print(f"\nResults saved to:")
    print(f"- CSV: {csv_path}")
    print(f"- TXT: {txt_path}")
    print(f"- Narrative Analysis: {narrative_path}")
    print(f"\nProcessed {len(results)} frames with temporal attention and text feedback enabled.")
    
    # Display sample narrative flow
    if len(results) >= 3:
        print(f"\nSample Narrative Flow:")
        print("-" * 40)
        for i in range(min(3, len(results))):
            r = results[i]
            print(f"Frame {r['frame_idx']:04d}: {r['caption']}")

if __name__ == "__main__":
    main()

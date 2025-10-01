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
    
    # Reset temporal memory for new video sequence
    model.reset_temporal_memory()
    
    results = []
    save_dir = os.path.join(cfg.inference.inf_save_dir, "temporal_captions")
    os.makedirs(save_dir, exist_ok=True)
    
    # Process frames in batches of sequential frames (like training)
    batch_size = cfg.inference.batch_size
    
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
                batch_audios.append(audio_tensor.squeeze(0))  # Remove batch dim for batching
                
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
            
        # Stack into batch tensors
        images_batch = torch.stack(batch_images).to(device)  # [B, C, H, W]
        audios_batch = torch.stack(batch_audios).to(device)  # [B, n_mfcc, T]
        
        # Process batch with temporal context (each frame can attend to previous frames in batch)
        with torch.no_grad():
            # Process each frame in the batch sequentially, building temporal context
            for i in range(len(batch_info)):
                # Current frame
                current_image = images_batch[i:i+1]  # [1, C, H, W]
                current_audio = audios_batch[i:i+1]  # [1, n_mfcc, T]
                
                # Generate caption with access to temporal memory from previous frames
                generated_ids = model.generate(
                    images=current_image,
                    audio=current_audio,
                    max_length=cfg.inference.max_length,
                    num_beams=cfg.inference.num_beams
                )
                
                # Decode caption
                caption = model.decoder.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Store results
                result_info = batch_info[i].copy()
                result_info['caption'] = caption
                results.append(result_info)
                
                print(f"Batch {batch_start//batch_size + 1}, Frame {i+1}/{len(batch_info)} "
                      f"({result_info['frame_name']}): {caption}")
                
                # Update temporal memory by running forward pass (this updates the memory buffer)
                if i < len(batch_info) - 1:  # Don't need to update memory on last frame of batch
                    # Run a forward pass to update temporal memory with the generated caption
                    # This ensures the temporal memory gets the actual output, not a dummy
                    _ = model.forward(
                        current_image,
                        caption,  # Use the generated caption as input for temporal memory
                        current_audio
                    )
    
    # Save results
    # Save as CSV for analysis
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "temporal_captions.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Save as text file
    txt_path = os.path.join(save_dir, "temporal_captions.txt")
    with open(txt_path, 'w') as f:
        for r in results:
            f.write(f"Frame {r['frame_idx']:04d} ({r['frame_name']}): {r['caption']}\n")
    
    print(f"\nResults saved to:")
    print(f"- CSV: {csv_path}")
    print(f"- TXT: {txt_path}")
    print(f"\nProcessed {len(results)} frames with temporal attention enabled.")

if __name__ == "__main__":
    main()

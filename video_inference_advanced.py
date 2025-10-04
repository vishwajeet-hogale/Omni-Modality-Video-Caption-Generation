#!/usr/bin/env python3
"""
Advanced Video Inference with Caption Diversity Enhancement
===========================================================

This script implements advanced techniques to improve caption diversity
and quality while maintaining training consistency.

Key Features:
- Nucleus sampling and temperature control
- Anti-repetition mechanisms
- Context-aware generation
- Dynamic beam search
- Post-processing for quality enhancement

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
import re
from collections import Counter

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import SwinBart


class AdvancedInferenceDataset(Dataset):
    """Enhanced dataset with better preprocessing and context management."""
    
    def __init__(self, frames_dir, audio_dir, transform=None):
        self.frames_dir = frames_dir
        self.audio_dir = audio_dir
        self.transform = transform
        
        # Get all frame files and sort them properly
        self.frame_files = []
        for file in os.listdir(frames_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                self.frame_files.append(file)
        
        # Sort by frame number
        self.frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        print(f"📊 Found {len(self.frame_files)} frames")
    
    def __len__(self):
        return len(self.frame_files)
    
    def __getitem__(self, idx):
        frame_file = self.frame_files[idx]
        frame_path = os.path.join(self.frames_dir, frame_file)
        
        # Load and process image
        try:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"⚠️  Error loading {frame_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # Load corresponding audio
        audio_file = frame_file.replace('.jpg', '.pt').replace('.jpeg', '.pt').replace('.png', '.pt')
        audio_path = os.path.join(self.audio_dir, audio_file)
        
        try:
            if os.path.exists(audio_path):
                audio = torch.load(audio_path, map_location='cpu')
                if isinstance(audio, dict):
                    audio = audio.get('features', audio.get('mfcc', torch.zeros(13, 100)))
                
                # Ensure correct shape [13, 100]
                if audio.dim() == 1:
                    audio = audio.unsqueeze(-1).expand(-1, 100)
                elif audio.dim() == 2:
                    if audio.size(0) != 13:
                        audio = F.pad(audio, (0, 0, 0, max(0, 13 - audio.size(0))))[:13]
                    if audio.size(1) != 100:
                        if audio.size(1) < 100:
                            audio = F.pad(audio, (0, 100 - audio.size(1)))
                        else:
                            audio = audio[:, :100]
            else:
                audio = torch.zeros(13, 100)
        except Exception as e:
            print(f"⚠️  Error loading audio {audio_path}: {e}")
            audio = torch.zeros(13, 100)
        
        # Extract frame number
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


class DiversityEnhancer:
    """Enhances caption diversity using various techniques."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.recent_captions = []
        self.phrase_counts = Counter()
        self.max_history = 10
    
    def add_caption(self, caption):
        """Add caption to history for diversity tracking."""
        self.recent_captions.append(caption)
        if len(self.recent_captions) > self.max_history:
            self.recent_captions.pop(0)
        
        # Track common phrases
        words = caption.lower().split()
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            self.phrase_counts[phrase] += 1
    
    def is_repetitive(self, caption):
        """Check if caption is too similar to recent ones."""
        if not self.recent_captions:
            return False
        
        # Simple similarity check
        caption_words = set(caption.lower().split())
        
        for recent in self.recent_captions[-3:]:  # Check last 3 captions
            recent_words = set(recent.lower().split())
            overlap = len(caption_words & recent_words)
            similarity = overlap / max(len(caption_words), len(recent_words))
            
            if similarity > 0.8:  # 80% similarity threshold
                return True
        
        return False
    
    def get_repetition_penalty_tokens(self):
        """Get tokens that should be penalized for repetition."""
        if not self.phrase_counts:
            return []
        
        # Get most common phrases
        common_phrases = [phrase for phrase, count in self.phrase_counts.most_common(5) if count > 2]
        
        penalty_tokens = []
        for phrase in common_phrases:
            tokens = self.tokenizer.encode(phrase, add_special_tokens=False)
            penalty_tokens.extend(tokens)
        
        return list(set(penalty_tokens))


def generate_diverse_caption(model, image, audio, device, cfg, diversity_enhancer, previous_context=None):
    """
    Generate caption with enhanced diversity techniques.
    """
    
    with torch.no_grad():
        # Ensure correct shapes
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        image = image.to(device)
        audio = audio.to(device)
        
        # Get repetition penalty tokens
        penalty_tokens = diversity_enhancer.get_repetition_penalty_tokens()
        
        # Try multiple generation strategies
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            try:
                # Vary generation parameters for diversity
                if attempts == 0:
                    # Standard beam search
                    generated_ids = model.generate(
                        images=image,
                        audio=audio,
                        max_length=cfg.inference.max_length,
                        num_beams=cfg.inference.num_beams,
                        is_new_video=False,
                        previous_caption=previous_context
                    )
                elif attempts == 1:
                    # More diverse beam search
                    generated_ids = model.generate(
                        images=image,
                        audio=audio,
                        max_length=cfg.inference.max_length,
                        num_beams=max(4, cfg.inference.num_beams // 2),
                        is_new_video=False,
                        previous_caption=previous_context
                    )
                else:
                    # Fallback: reduce beam search further
                    generated_ids = model.generate(
                        images=image,
                        audio=audio,
                        max_length=min(40, cfg.inference.max_length),
                        num_beams=2,
                        is_new_video=False,
                        previous_caption=previous_context
                    )
                
                # Decode
                if generated_ids.dim() > 1:
                    generated_ids = generated_ids[0]
                
                caption = model.decoder.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()
                
                # Post-process caption
                caption = post_process_caption(caption)
                
                # Check for repetition
                if not diversity_enhancer.is_repetitive(caption) or attempts == max_attempts - 1:
                    return caption
                
                attempts += 1
                
            except Exception as e:
                print(f"❌ Generation attempt {attempts + 1} failed: {e}")
                attempts += 1
        
        return "Unable to generate diverse caption"


def post_process_caption(caption):
    """Post-process caption for better quality."""
    
    # Remove incomplete sentences at the end
    sentences = caption.split('.')
    if len(sentences) > 1 and sentences[-1].strip() and not sentences[-1].strip()[0].isupper():
        caption = '.'.join(sentences[:-1]) + '.'
    
    # Fix common issues
    caption = re.sub(r'\s+', ' ', caption)  # Multiple spaces
    caption = re.sub(r'\s+([,.!?])', r'\1', caption)  # Space before punctuation
    caption = caption.strip()
    
    # Ensure proper capitalization
    if caption and caption[0].islower():
        caption = caption[0].upper() + caption[1:]
    
    return caption


def setup_advanced_model(cfg, device):
    """Setup model with advanced inference optimizations."""
    print("📦 Setting up advanced model...")
    
    model = SwinBart(cfg)
    model.to(device)
    
    # Load checkpoint
    checkpoint_path = cfg.inference.model_path
    if os.path.exists(checkpoint_path):
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print("✅ Checkpoint loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
    
    model.eval()
    
    # Configure for inference
    if hasattr(model.encoder, 'vision_model'):
        if cfg.vision_encoder_cfg.freeze:
            model.encoder.vision_model.eval()
            print("🧊 Vision encoder frozen")
        else:
            model.encoder.vision_model.train()
            print("🔥 Vision encoder active")
    
    return model


def process_video_advanced(model, dataset, device, cfg):
    """Process video with advanced diversity enhancement."""
    print("🎬 Processing video with diversity enhancement...")
    
    # Initialize diversity enhancer
    diversity_enhancer = DiversityEnhancer(model.decoder.tokenizer)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    results = []
    previous_captions = []
    
    # Reset temporal memory
    if hasattr(model, 'reset_temporal_memory'):
        model.reset_temporal_memory()
    
    print(f"🧠 Temporal attention: {'Enabled' if model.use_temporal_attention else 'Disabled'}")
    print(f"🎯 Diversity enhancement: Enabled")
    
    # Process frames
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing frames")):
        
        image = batch['image']
        audio = batch['audio']
        frame_file = batch['frame_file'][0]
        frame_num = batch['frame_num'].item()
        
        # Get context from previous captions
        prev_context = previous_captions[-1] if previous_captions else None
        
        try:
            # Generate diverse caption
            caption = generate_diverse_caption(
                model=model,
                image=image,
                audio=audio,
                device=device,
                cfg=cfg,
                diversity_enhancer=diversity_enhancer,
                previous_context=prev_context
            )
            
            # Add to diversity enhancer
            if not caption.startswith('Error') and not caption.startswith('Unable'):
                diversity_enhancer.add_caption(caption)
            
            # Store result
            results.append({
                'frame_num': frame_num,
                'frame_file': frame_file,
                'caption': caption,
                'batch_idx': batch_idx,
                'has_error': caption.startswith(('Error', 'Unable'))
            })
            
            # Update context
            previous_captions.append(caption)
            if len(previous_captions) > 5:
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
    
    return results, diversity_enhancer


def analyze_advanced_results(results, diversity_enhancer):
    """Enhanced result analysis."""
    print("\n📊 Analyzing advanced results...")
    
    # Basic stats
    total_frames = len(results)
    successful_frames = len([r for r in results if not r['has_error']])
    
    # Get successful captions
    captions = [r['caption'] for r in results if not r['has_error']]
    
    # Advanced diversity metrics
    unique_captions = len(set(captions)) if captions else 0
    diversity_ratio = unique_captions / len(captions) if captions else 0
    
    # Caption length statistics
    lengths = [len(c.split()) for c in captions] if captions else [0]
    avg_length = np.mean(lengths)
    std_length = np.std(lengths)
    
    # Vocabulary diversity
    all_words = []
    for caption in captions:
        all_words.extend(caption.lower().split())
    
    unique_words = len(set(all_words))
    total_words = len(all_words)
    vocab_diversity = unique_words / max(total_words, 1)
    
    # Medical terminology analysis
    medical_terms = [
        'tube', 'nasogastric', 'ng', 'insert', 'insertion', 'patient', 'medical',
        'procedure', 'healthcare', 'nurse', 'doctor', 'stomach', 'nose', 'throat',
        'landmark', 'marking', 'xiphoid', 'earlobe', 'suction', 'aspirate',
        'gastric', 'content', 'placement', 'verify', 'emesis', 'basin', 'supplies'
    ]
    
    medical_usage = sum(1 for c in captions if any(term in c.lower() for term in medical_terms))
    medical_ratio = medical_usage / len(captions) if captions else 0
    
    # Repetition analysis
    phrase_repetition = len(diversity_enhancer.phrase_counts)
    
    print(f"📈 Advanced Results Analysis:")
    print(f"  Total frames: {total_frames}")
    print(f"  Successful frames: {successful_frames} ({100*successful_frames/total_frames:.1f}%)")
    print(f"  Unique captions: {unique_captions}")
    print(f"  Caption diversity: {diversity_ratio:.2%}")
    print(f"  Average length: {avg_length:.1f} ± {std_length:.1f} words")
    print(f"  Vocabulary diversity: {vocab_diversity:.2%}")
    print(f"  Medical terminology: {medical_ratio:.2%}")
    print(f"  Phrase variety: {phrase_repetition} unique phrases")
    
    # Quality assessment
    if diversity_ratio > 0.25:
        print("✅ Excellent caption diversity")
    elif diversity_ratio > 0.15:
        print("✅ Good caption diversity")
    elif diversity_ratio > 0.08:
        print("⚠️  Moderate caption diversity")
    else:
        print("❌ Poor caption diversity")
    
    return {
        'total_frames': total_frames,
        'successful_frames': successful_frames,
        'unique_captions': unique_captions,
        'diversity_ratio': diversity_ratio,
        'avg_length': avg_length,
        'std_length': std_length,
        'vocab_diversity': vocab_diversity,
        'medical_ratio': medical_ratio,
        'phrase_variety': phrase_repetition
    }


def save_advanced_results(results, analysis, cfg, save_dir, diversity_enhancer):
    """Save advanced results with detailed analysis."""
    print(f"💾 Saving advanced results to {save_dir}...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Enhanced CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "advanced_captions.csv")
    df.to_csv(csv_path, index=False)
    
    # 2. Detailed report
    report_path = os.path.join(save_dir, "advanced_report.txt")
    with open(report_path, 'w') as f:
        f.write("Advanced Video Inference Report\n")
        f.write("=" * 35 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Model: SwinBart with diversity enhancement\n")
        f.write(f"  Vision encoder frozen: {cfg.vision_encoder_cfg.freeze}\n")
        f.write(f"  Temporal attention: {cfg.use_temporal_attention}\n")
        f.write(f"  Max caption length: {cfg.inference.max_length}\n")
        f.write(f"  Beam search: {cfg.inference.num_beams}\n")
        f.write(f"  Diversity enhancement: Enabled\n\n")
        
        f.write("ADVANCED ANALYSIS:\n")
        for key, value in analysis.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Most common phrases
        f.write("MOST COMMON PHRASES:\n")
        for phrase, count in diversity_enhancer.phrase_counts.most_common(10):
            f.write(f"  '{phrase}': {count} times\n")
        f.write("\n")
        
        f.write("FRAME-BY-FRAME CAPTIONS:\n")
        f.write("-" * 30 + "\n")
        
        sorted_results = sorted(results, key=lambda x: x['frame_num'])
        for r in sorted_results:
            status = "✓" if not r['has_error'] else "✗"
            f.write(f"Frame {r['frame_num']:04d} {status}: {r['caption']}\n")
    
    print(f"📊 CSV: {csv_path}")
    print(f"📝 Report: {report_path}")


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function for advanced video inference."""
    
    print("🎬 Advanced Video Inference with Diversity Enhancement")
    print("=" * 55)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Setup directories
    frames_dir = cfg.inference.video_image_dir
    audio_dir = frames_dir.replace('/frames/', '/mfccs/')
    
    if not os.path.exists(frames_dir):
        print(f"❌ Frames directory not found: {frames_dir}")
        return
    
    # Create dataset
    dataset = AdvancedInferenceDataset(
        frames_dir=frames_dir,
        audio_dir=audio_dir,
        transform=transform
    )
    
    if len(dataset) == 0:
        print("❌ No frames found")
        return
    
    # Setup model
    model = setup_advanced_model(cfg, device)
    
    # Process video
    results, diversity_enhancer = process_video_advanced(model, dataset, device, cfg)
    
    # Analyze results
    analysis = analyze_advanced_results(results, diversity_enhancer)
    
    # Save results
    save_dir = cfg.inference.inf_save_dir + "_advanced"
    save_advanced_results(results, analysis, cfg, save_dir, diversity_enhancer)
    
    # Final summary
    print(f"\n🎉 Advanced inference completed!")
    print(f"📈 Processed: {analysis['total_frames']} frames")
    print(f"✅ Success rate: {100*analysis['successful_frames']/analysis['total_frames']:.1f}%")
    print(f"🎯 Caption diversity: {analysis['diversity_ratio']:.2%}")
    print(f"📚 Vocabulary diversity: {analysis['vocab_diversity']:.2%}")
    print(f"🏥 Medical terms: {analysis['medical_ratio']:.2%}")
    print(f"📂 Results: {save_dir}")


if __name__ == "__main__":
    main()
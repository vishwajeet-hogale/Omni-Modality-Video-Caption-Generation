import torch
import hydra
from omegaconf import DictConfig
from models.model import SwinBart
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import json
from collections import Counter
import math
from typing import List, Tuple, Dict

# Text normalization and evaluation utilities
def _normalize_text(s: str) -> str:
    """Normalize text for evaluation"""
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[""]", '"', s)
    s = re.sub(r"[\u2018\u2019]", "'", s)
    return s

def _tok(s: str):
    """Tokenize text"""
    return _normalize_text(s).split()

def _ngram_counts(tokens, n):
    """Count n-grams in token sequence"""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def _corpus_bleu(hypotheses_tokens, references_tokens_list, max_n=4):
    """Calculate BLEU score with add-1 smoothing"""
    precisions = []
    for n in range(1, max_n + 1):
        overlap = 0
        total = 0
        for hyp, refs in zip(hypotheses_tokens, references_tokens_list):
            hyp_ngrams = _ngram_counts(hyp, n)
            max_ref_counts = Counter()
            for r in refs:
                ref_ngrams = _ngram_counts(r, n)
                for ng, cnt in ref_ngrams.items():
                    if cnt > max_ref_counts[ng]:
                        max_ref_counts[ng] = cnt
            for ng, cnt in hyp_ngrams.items():
                overlap += min(cnt, max_ref_counts.get(ng, 0))
            total += sum(hyp_ngrams.values())
        precisions.append((overlap + 1) / (total + 1))  # add-1 smoothing

    hyp_len_total = sum(len(h) for h in hypotheses_tokens)
    ref_len_total = 0
    for hyp, refs in zip(hypotheses_tokens, references_tokens_list):
        hyp_len = len(hyp)
        closest = min((abs(len(r) - hyp_len), len(r)) for r in refs)
        ref_len_total += closest[1]

    if hyp_len_total == 0:
        bp = 0.0
    elif hyp_len_total > ref_len_total:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len_total / hyp_len_total)

    log_prec = sum(math.log(p) for p in precisions) / max_n
    bleu = bp * math.exp(log_prec)
    return bleu, bp, precisions

def _lcs_len(a, b):
    """Calculate Longest Common Subsequence length"""
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        ai = a[i-1]
        row_i = dp[i]
        row_im1 = dp[i-1]
        for j in range(1, n+1):
            if ai == b[j-1]:
                row_i[j] = row_im1[j-1] + 1
            else:
                row_i[j] = max(row_im1[j], row_i[j-1])
    return dp[m][n]

def _rouge_l_f1(hyp_tokens, ref_tokens):
    """Calculate ROUGE-L F1 score"""
    if not hyp_tokens and not ref_tokens:
        return 1.0
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(hyp_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    p = lcs / len(hyp_tokens)
    r = lcs / len(ref_tokens)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def _parse_frame_idx(filename: str):
    """Parse frame index from filename"""
    base = os.path.basename(filename)
    m = re.search(r'frame[_-]?(\d+)', base, flags=re.IGNORECASE)
    if not m:
        m2 = re.findall(r'(\d+)', base)
        if not m2:
            return None
        return int(m2[-1])
    return int(m.group(1))

def get_frame_files(video_dir: str) -> List[Tuple[str, int]]:
    """Get all frame files sorted by frame index"""
    valid_exts = {'.jpg', '.jpeg', '.png'}
    frame_files = []
    
    for filename in os.listdir(video_dir):
        if os.path.splitext(filename)[-1].lower() in valid_exts:
            frame_idx = _parse_frame_idx(filename)
            if frame_idx is not None:
                frame_files.append((filename, frame_idx))
    
    # Sort by frame index
    frame_files.sort(key=lambda x: x[1])
    return frame_files

def process_video_frames_sequential(cfg, model, transform, device, video_dir: str, 
                                  ground_truth_captions: List[str] = None) -> Dict:
    """
    Process video frames sequentially with temporal attention
    
    Key changes from original inference:
    1. Sequential processing: Process frames one by one to maintain temporal context
    2. Temporal memory: Reset memory at start, let it build up during processing
    3. Batch processing: Process frames in batches of 8 (like training) but maintain order
    4. Memory management: Use temporal attention and memory throughout
    """
    
    # Reset temporal memory for new video
    model.reset_temporal_memory()
    
    # Get all frame files sorted by frame index
    frame_files = get_frame_files(video_dir)
    if not frame_files:
        print(f"No valid frame files found in {video_dir}")
        return {"predictions": [], "metrics": {}}
    
    print(f"Processing {len(frame_files)} frames from {video_dir}")
    
    # Process frames in batches of 8 (like training)
    batch_size = cfg.trainer.batch_size  # Use training batch size
    predictions = []
    pred_rows = []
    
    # Process frames in sequential batches
    for batch_start in tqdm(range(0, len(frame_files), batch_size), desc="Processing frame batches"):
        batch_end = min(batch_start + batch_size, len(frame_files))
        batch_frames = frame_files[batch_start:batch_end]
        
        # Prepare batch data
        batch_images = []
        batch_audio = []
        batch_frame_info = []
        
        for filename, frame_idx in batch_frames:
            try:
                # Load image
                image_path = os.path.join(video_dir, filename)
                image = Image.open(image_path).convert("RGB")
                image_tensor = transform(image)
                batch_images.append(image_tensor)
                
                # Load corresponding audio
                audio_path = image_path.replace(f"{os.sep}frames{os.sep}", f"{os.sep}mfccs{os.sep}")
                audio_path = audio_path.replace(f"{os.sep}frames_dup{os.sep}", f"{os.sep}mfccs{os.sep}")
                audio_path = os.path.splitext(audio_path)[0] + ".npy"
                
                if os.path.exists(audio_path):
                    audio_arr = np.load(audio_path)
                    audio_tensor = torch.from_numpy(audio_arr).float()
                    if audio_tensor.dim() == 2:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    elif audio_tensor.dim() == 3 and audio_tensor.shape[0] != 1:
                        audio_tensor = audio_tensor[:1]
                    batch_audio.append(audio_tensor)
                else:
                    print(f"Warning: Audio file not found: {audio_path}")
                    # Create dummy audio tensor
                    batch_audio.append(torch.zeros(1, 13, 100))  # dummy MFCC
                
                batch_frame_info.append((filename, frame_idx))
                
            except Exception as e:
                print(f"Error loading frame {filename}: {e}")
                continue
        
        if not batch_images:
            continue
            
        # Stack tensors for batch processing
        images_tensor = torch.stack(batch_images).to(device)
        audio_tensor = torch.stack(batch_audio).to(device)
        
        # Process batch with temporal attention
        with torch.no_grad():
            # Generate captions for the batch
            # Note: The model's forward method handles temporal attention internally
            generated_ids = model.generate(
                images=images_tensor,
                audio=audio_tensor,
                max_length=cfg.inference.max_length,
                num_beams=cfg.inference.num_beams
            )
            
            # Decode captions
            for i, (filename, frame_idx) in enumerate(batch_frame_info):
                if i < len(generated_ids):
                    caption = model.decoder.tokenizer.decode(
                        generated_ids[i], skip_special_tokens=True
                    )
                    predictions.append(caption)
                    
                    # Get reference caption if available
                    ref = ""
                    if ground_truth_captions and frame_idx < len(ground_truth_captions):
                        ref = ground_truth_captions[frame_idx]
                    
                    pred_rows.append({
                        "frame_idx": frame_idx,
                        "image": filename,
                        "prediction": caption,
                        "reference": ref
                    })
    
    # Calculate evaluation metrics if ground truth is available
    metrics = {}
    if ground_truth_captions and pred_rows:
        # Filter valid predictions
        valid_eval = [row for row in pred_rows if row["reference"]]
        if valid_eval:
            hyps = [_tok(row["prediction"]) for row in valid_eval]
            refs = [[_tok(row["reference"])] for row in valid_eval]
            
            bleu, bp, precisions = _corpus_bleu(hyps, refs, max_n=4)
            
            # ROUGE-L
            rouge_sum = 0.0
            for hyp_tokens, ref_tokens in zip(hyps, (r[0] for r in refs)):
                rouge_sum += _rouge_l_f1(hyp_tokens, ref_tokens)
            rouge_avg = rouge_sum / len(hyps)
            
            metrics = {
                "matched": len(hyps),
                "bleu": float(bleu),
                "bleu_precisions": [float(p) for p in precisions],
                "brevity_penalty": float(bp),
                "rougeL_F1": float(rouge_avg)
            }
    
    return {
        "predictions": predictions,
        "pred_rows": pred_rows,
        "metrics": metrics
    }

def predict_and_annotate_video_sequential(cfg, model, transform, device):
    """
    Main function to process video with sequential frame-by-frame inference
    
    Key changes:
    1. Sequential processing instead of parallel
    2. Temporal memory management
    3. Batch processing while maintaining temporal order
    4. Memory reset between videos
    """
    
    video_dir = cfg.inference.video_image_dir
    save_dir = os.path.join(cfg.inference.inf_save_dir, "captioned")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load ground truth captions
    ground_truth_captions = None
    gt_csv_path = os.path.join(cfg.data.captions_dir, os.path.basename(video_dir) + ".csv")
    if os.path.exists(gt_csv_path):
        gt_df = pd.read_csv(gt_csv_path)
        if "caption" in gt_df.columns:
            ground_truth_captions = gt_df["caption"].tolist()
            print(f"Loaded {len(ground_truth_captions)} ground truth captions")
    
    # Process video frames sequentially
    results = process_video_frames_sequential(
        cfg, model, transform, device, video_dir, ground_truth_captions
    )
    
    predictions = results["predictions"]
    pred_rows = results["pred_rows"]
    metrics = results["metrics"]
    
    # Save annotated images
    frame_files = get_frame_files(video_dir)
    for i, (filename, frame_idx) in enumerate(frame_files):
        if i < len(predictions):
            try:
                image_path = os.path.join(video_dir, filename)
                image = Image.open(image_path).convert("RGB")
                
                # Draw caption on image
                annotated_image = image.copy()
                draw = ImageDraw.Draw(annotated_image)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except IOError:
                    font = ImageFont.load_default()
                
                caption = predictions[i]
                margin = 10
                text_pos = (margin, annotated_image.height - 30)
                draw.rectangle([text_pos, (annotated_image.width - margin, annotated_image.height - margin)],
                               fill=(0, 0, 0, 150))
                draw.text(text_pos, caption, font=font, fill=(255, 255, 255))
                
                # Save annotated image
                save_path = os.path.join(save_dir, filename)
                annotated_image.save(save_path)
                
            except Exception as e:
                print(f"Error annotating {filename}: {e}")
    
    # Save results and metrics
    eval_dir = os.path.join(cfg.inference.inf_save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save predictions
    if pred_rows:
        pred_df = pd.DataFrame(pred_rows)
        pred_csv_path = os.path.join(eval_dir, "predictions_sequential.csv")
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"Saved predictions to: {pred_csv_path}")
    
    # Save metrics
    if metrics:
        metrics_path = os.path.join(eval_dir, "metrics_sequential.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print("\n=== Sequential Processing Evaluation ===")
        print(f"Matched examples: {metrics['matched']}")
        print(f"BLEU-4: {metrics['bleu']:.4f}")
        print(f"ROUGE-L F1: {metrics['rougeL_F1']:.4f}")
        print(f"Saved metrics to: {metrics_path}")
    
    return metrics

@hydra.main(config_path="configs", config_name="default")
def inference_sequential(cfg: DictConfig):
    """
    Sequential video inference with temporal attention
    
    Key differences from original inference:
    1. Frame-by-frame processing to maintain temporal context
    2. Temporal memory and attention throughout processing
    3. Batch processing while preserving temporal order
    4. Memory reset between videos
    """
    
    # Set device
    device = torch.device(cfg.trainer.device)
    print(f"Using device: {device}")
    
    # Initialize and load model
    model = SwinBart(cfg).to(device)
    model.load_state_dict(torch.load(cfg.inference.model_path, map_location=device), strict=False)
    model.eval()
    print("Model loaded successfully")
    
    # Check if temporal attention is enabled
    if hasattr(model, 'use_temporal_attention') and model.use_temporal_attention:
        print("Temporal attention enabled - processing frames sequentially")
    else:
        print("Warning: Temporal attention not enabled in model")
    
    # Image preprocessing (same as training)
    transform = transforms.Compose([
        transforms.Resize((cfg.inference.image_size, cfg.inference.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process video with sequential inference
    metrics = predict_and_annotate_video_sequential(cfg, model, transform, device)
    
    print("Sequential inference completed!")
    return metrics

if __name__ == "__main__":
    inference_sequential()

import torch
import torch.nn.functional as F
import os
import json
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import librosa
import numpy as np
from tqdm import tqdm

from models.model import SwinBart
import hydra
from omegaconf import DictConfig

class MultimodalDiffusionInference:
    """Inference class for multimodal diffusion model"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self._get_device()
        
        # Initialize model
        self.model = SwinBart(cfg)
        
        # Load checkpoint if provided
        if cfg.inference.model_path and os.path.exists(cfg.inference.model_path):
            print(f"Loading model from {cfg.inference.model_path}")
            checkpoint = torch.load(cfg.inference.model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("Model loaded successfully!")
        else:
            print("Warning: No model checkpoint found, using random weights")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize tokenizer for text processing
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def _get_device(self):
        """Get the best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def load_audio(self, audio_path):
        """Load and preprocess audio"""
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfccs = torch.tensor(mfccs, dtype=torch.float32)
            
            # Pad or truncate to fixed length
            target_length = 100  # Adjust based on your needs
            if mfccs.shape[1] > target_length:
                mfccs = mfccs[:, :target_length]
            else:
                padding = target_length - mfccs.shape[1]
                mfccs = F.pad(mfccs, (0, padding))
            
            return mfccs.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def generate_caption_standard(self, images, audio, max_length=60):
        """Generate caption using standard model"""
        with torch.no_grad():
            generated_ids = self.model.generate(
                images, audio, 
                max_length=max_length, 
                num_beams=4,
                is_new_video=True
            )
            
            # Decode tokens to text
            captions = []
            for ids in generated_ids:
                caption = self.tokenizer.decode(ids, skip_special_tokens=True)
                captions.append(caption)
            
            return captions
    
    def generate_caption_diffusion(self, images, audio, max_length=60):
        """Generate caption using diffusion model"""
        if not self.cfg.use_multimodal_diffusion:
            print("Warning: Multi-modal diffusion not enabled, falling back to standard generation")
            return self.generate_caption_standard(images, audio, max_length)
        
        with torch.no_grad():
            try:
                generated_tokens = self.model.generate_with_diffusion(
                    images, audio, max_length=max_length
                )
                
                # Decode tokens to text
                captions = []
                for tokens in generated_tokens:
                    caption = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    captions.append(caption)
                
                return captions
            except Exception as e:
                print(f"Error in diffusion generation: {e}")
                return self.generate_caption_standard(images, audio, max_length)
    
    def generate_caption_masked(self, images, audio, text, mask_prob=0.15):
        """Generate caption using masked language modeling"""
        if not self.cfg.use_multimodal_diffusion:
            print("Warning: Multi-modal diffusion not enabled, falling back to standard generation")
            return self.generate_caption_standard(images, audio)
        
        with torch.no_grad():
            try:
                predictions, mask_positions = self.model.generate_with_masking(
                    images, audio, text, mask_prob
                )
                
                # Decode predictions
                captions = []
                for pred in predictions:
                    caption = self.tokenizer.decode(pred, skip_special_tokens=True)
                    captions.append(caption)
                
                return captions, mask_positions
            except Exception as e:
                print(f"Error in masked LM generation: {e}")
                return self.generate_caption_standard(images, audio)
    
    def process_video_sequence(self, video_dir, output_dir, method="standard"):
        """Process a video sequence and generate captions"""
        print(f"Processing video directory: {video_dir}")
        print(f"Using method: {method}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        
        if not image_files:
            print(f"No image files found in {video_dir}")
            return
        
        print(f"Found {len(image_files)} images")
        
        results = []
        
        for i, image_file in enumerate(tqdm(image_files, desc="Processing frames")):
            image_path = os.path.join(video_dir, image_file)
            
            # Load image
            image = self.load_image(image_path)
            if image is None:
                continue
            
            # For now, create dummy audio (you can replace this with actual audio loading)
            audio = torch.randn(1, 13, 100)  # Dummy MFCC features
            
            # Move to device
            image = image.to(self.device)
            audio = audio.to(self.device)
            
            try:
                # Generate caption based on method
                if method == "standard":
                    captions = self.generate_caption_standard(image, audio)
                elif method == "diffusion":
                    captions = self.generate_caption_diffusion(image, audio)
                elif method == "masked":
                    # For masked LM, we need text input
                    dummy_text = torch.randint(0, 1000, (1, 50))  # Dummy text tokens
                    captions, mask_positions = self.generate_caption_masked(image, audio, dummy_text)
                else:
                    captions = self.generate_caption_standard(image, audio)
                
                # Store result
                result = {
                    'frame': image_file,
                    'caption': captions[0] if captions else "No caption generated",
                    'method': method
                }
                results.append(result)
                
                print(f"Frame {i+1}/{len(image_files)}: {captions[0] if captions else 'No caption'}")
                
            except Exception as e:
                print(f"Error processing frame {image_file}: {e}")
                results.append({
                    'frame': image_file,
                    'caption': "Error generating caption",
                    'method': method
                })
        
        # Save results
        self.save_results(results, output_dir, method)
        
        return results
    
    def save_results(self, results, output_dir, method):
        """Save inference results"""
        # Save as CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, f"captions_{method}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Save as JSON
        json_path = os.path.join(output_dir, f"captions_{method}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {json_path}")
        
        # Save summary
        summary_path = os.path.join(output_dir, f"summary_{method}.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Multimodal Diffusion Inference Results\n")
            f.write(f"Method: {method}\n")
            f.write(f"Total frames: {len(results)}\n")
            f.write(f"Successful generations: {len([r for r in results if 'Error' not in r['caption']])}\n")
            f.write(f"Failed generations: {len([r for r in results if 'Error' in r['caption']])}\n")
            
            # Calculate narrative coherence
            connective_words = ['then', 'next', 'after', 'while', 'during', 'meanwhile', 'subsequently']
            connective_count = sum(1 for r in results if any(word in r['caption'].lower() for word in connective_words))
            if len(results) > 0:
                f.write(f"Narrative coherence: {connective_count/len(results)*100:.1f}%\n")
            else:
                f.write(f"Narrative coherence: 0.0% (no results)\n")
        
        print(f"Summary saved to {summary_path}")

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    """Main inference function"""
    
    # Initialize inference class
    inference = MultimodalDiffusionInference(cfg)
    
    # Get video directory
    video_dir = cfg.inference.video_image_dir
    if not os.path.exists(video_dir):
        print(f"Error: Video directory not found: {video_dir}")
        return
    
    # Get output directory
    output_dir = cfg.inference.inf_save_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing video directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    
    # Test different methods
    methods = ["standard", "diffusion", "masked"]
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing {method} method")
        print(f"{'='*50}")
        
        try:
            results = inference.process_video_sequence(
                video_dir, 
                output_dir, 
                method=method
            )
            print(f"✅ {method} method completed successfully")
        except Exception as e:
            print(f"❌ {method} method failed: {e}")
    
    print(f"\n🎉 Inference completed! Check results in {output_dir}")

if __name__ == "__main__":
    main()

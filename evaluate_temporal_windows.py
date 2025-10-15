#!/usr/bin/env python3
"""
Script to evaluate the model on different temporal windows
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import json
from datetime import datetime

from models.model import SwinBart
from data.datamodule import VideoDataModule
from train_multimodal_diffusion import MultimodalDiffusionTrainer

class TemporalWindowEvaluator:
    """Evaluator for different temporal window sizes"""
    
    def __init__(self, base_cfg):
        self.base_cfg = base_cfg
        self.results = {}
        
    def evaluate_temporal_window(self, temporal_window_size, max_caption_context):
        """Evaluate model with specific temporal window configuration"""
        print(f"\n{'='*60}")
        print(f"Evaluating temporal window size: {temporal_window_size}")
        print(f"Max caption context: {max_caption_context}")
        print(f"{'='*60}")
        
        # Create a copy of config with new temporal settings
        cfg = OmegaConf.create(self.base_cfg)
        cfg.model.temporal_window_size = temporal_window_size
        cfg.model.max_caption_context = max_caption_context
        cfg.temporal_memory_length = temporal_window_size
        
        # Quick testing settings to prevent laptop overheating
        cfg.trainer.batch_size = 2  # Very small batch size
        cfg.trainer.epochs = 2  # Only 2 epochs for quick testing
        cfg.data.batch_size = 2  # Match data batch size
        
        # Set up logging for this experiment with LoRA suffix
        lora_suffix = "_lora" if cfg.vision_encoder_cfg.lora else "_wo_lora"
        exp_name = f"temporal_eval_win{temporal_window_size}_ctx{max_caption_context}{lora_suffix}"
        cfg.trainer.exp_name = exp_name
        cfg.trainer.output_dir = f"./outputs/{datetime.now().strftime('%Y-%m-%d')}/{exp_name}"
        
        # Initialize data module
        data_module = VideoDataModule(cfg)
        
        # Initialize model
        model = MultimodalDiffusionTrainer(cfg)
        
        # Set up trainer for evaluation with LoRA-aware checkpointing
        logger = TensorBoardLogger(
            save_dir=cfg.trainer.output_dir,
            name=cfg.trainer.exp_name,
            version=None
        )
        
        # Set up checkpoint callback with LoRA suffix
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.trainer.output_dir,
            filename=f'best_model{lora_suffix}',
            save_top_k=1,
            monitor='val/loss',
            mode='min',
            save_last=True
        )
        
        # Force CPU usage to avoid disk space issues with MPS
        accelerator = "cpu"
        devices = "auto"
        
        trainer = L.Trainer(
            max_epochs=cfg.trainer.epochs,
            accelerator=accelerator,
            devices=devices,
            precision=cfg.trainer.precision,
            logger=logger,
            callbacks=[checkpoint_callback],
            gradient_clip_val=cfg.trainer.gradient_clip_val,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            enable_progress_bar=True
        )
        
        # Quick evaluation without training (just test compatibility)
        try:
            print("🔍 Testing temporal window compatibility (no training)...")
            
            # Get a small batch to test
            data_module.setup('test')
            test_loader = data_module.test_dataloader()
            batch = next(iter(test_loader))
            
            # Test forward pass with different temporal window
            model.eval()
            with torch.no_grad():
                images, captions, audio = batch["images"], batch["captions"], batch["audio"]
                
                # Test if temporal window works
                outputs = model(images, captions, audio, is_new_video=True)
                loss = outputs.loss
                
                print(f"✅ Temporal window {temporal_window_size} works! Loss: {loss.item():.4f}")
                
                # Store results
                test_results = [{'test/loss': loss.item(), 'test/compatibility': 'success'}]
            
            # Store results
            self.results[f"win{temporal_window_size}_ctx{max_caption_context}"] = {
                'temporal_window_size': temporal_window_size,
                'max_caption_context': max_caption_context,
                'test_results': test_results,
                'config': dict(cfg)
            }
            
            print(f"✅ Completed evaluation for window size {temporal_window_size}")
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            self.results[f"win{temporal_window_size}_ctx{max_caption_context}"] = {
                'temporal_window_size': temporal_window_size,
                'max_caption_context': max_caption_context,
                'error': str(e)
            }
    
    def run_evaluation(self, temporal_windows, caption_contexts):
        """Run evaluation on multiple temporal window configurations"""
        print("Starting temporal window evaluation...")
        print(f"Temporal windows: {temporal_windows}")
        print(f"Caption contexts: {caption_contexts}")
        print(f"Batch size: {self.base_cfg.trainer.batch_size}")
        print(f"Learning rate: {self.base_cfg.trainer.lr}")
        
        for window_size in temporal_windows:
            for caption_context in caption_contexts:
                self.evaluate_temporal_window(window_size, caption_context)
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save evaluation results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        lora_suffix = "_lora" if self.base_cfg.vision_encoder_cfg.lora else "_wo_lora"
        results_file = f"./outputs/temporal_evaluation_results{lora_suffix}_{timestamp}.json"
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📊 Results saved to: {results_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        for config_name, result in self.results.items():
            if 'error' in result:
                print(f"❌ {config_name}: ERROR - {result['error']}")
            else:
                test_loss = result['test_results'][0]['test/loss'] if result['test_results'] else 'N/A'
                print(f"✅ {config_name}: Test Loss = {test_loss}")

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    """Main evaluation function"""
    
    # Ultra-simplified temporal window configurations for quick testing
    # Only 3 configurations to prevent laptop overheating
    temporal_windows = [
        5,   # Very short-term (5 frames ≈ 0.17s)
        16,  # Medium-term (16 frames ≈ 0.53s) - common in video analysis
    ]
    
    # Caption context lengths for quick testing
    caption_contexts = [
        5,   # Short context
        15,  # Long context (current default)
    ]
    
    print("🎯 Industry-Standard Temporal Window Evaluation")
    print("="*60)
    print("Temporal Windows (frames):", temporal_windows)
    print("Caption Contexts:", caption_contexts)
    print("Total Configurations:", len(temporal_windows) * len(caption_contexts))
    print("="*60)
    
    # Initialize evaluator
    evaluator = TemporalWindowEvaluator(cfg)
    
    # Run evaluation
    evaluator.run_evaluation(temporal_windows, caption_contexts)

if __name__ == "__main__":
    main()

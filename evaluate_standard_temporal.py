#!/usr/bin/env python3
"""
Industry-Standard Temporal Window Evaluation
Based on common practices in video understanding research
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
from models.train_lightning import TrainerModule
from data.datamodule import VideoDataModule

class StandardTemporalEvaluator:
    """Evaluator for industry-standard temporal window sizes"""
    
    def __init__(self, base_cfg):
        self.base_cfg = base_cfg
        self.results = {}
        
        # Ultra-simplified temporal configurations for quick LoRA testing
        # Only 4 configurations to prevent laptop overheating
        self.standard_configs = [
            # Short-term (common in real-time applications)
            {"temporal_window": 3, "caption_context": 3, "name": "ultra_short"},
            {"temporal_window": 5, "caption_context": 5, "name": "very_short"},
            {"temporal_window": 8, "caption_context": 6, "name": "short_8"},
            
            # Medium-term (most common in video analysis)
            {"temporal_window": 10, "caption_context": 8, "name": "short"},
            {"temporal_window": 12, "caption_context": 9, "name": "short_12"},
            {"temporal_window": 16, "caption_context": 10, "name": "medium"},
            {"temporal_window": 24, "caption_context": 12, "name": "medium_24"},
            {"temporal_window": 32, "caption_context": 16, "name": "medium_32"},
            
            # Longer context (for higher-memory machines)
            {"temporal_window": 48, "caption_context": 20, "name": "long_48"},
            {"temporal_window": 64, "caption_context": 24, "name": "long_64"},
        ]
        
    def evaluate_standard_config(self, config):
        """Evaluate model with standard temporal configuration"""
        temporal_window = config["temporal_window"]
        caption_context = config["caption_context"]
        name = config["name"]
        
        print(f"\n{'='*60}")
        print(f"Evaluating {name} configuration:")
        print(f"  Temporal window: {temporal_window} frames")
        print(f"  Caption context: {caption_context}")
        print(f"  Time span: ~{temporal_window/30:.2f}s at 30fps")
        print(f"{'='*60}")
        
        # Create a copy of config with new temporal settings
        cfg = OmegaConf.create(self.base_cfg)
        cfg.model.temporal_window_size = temporal_window
        cfg.model.max_caption_context = caption_context
        cfg.temporal_memory_length = temporal_window
        
        # Ultra-quick LoRA testing settings to prevent laptop overheating
        cfg.trainer.batch_size = 1  # Ultra-small batch size
        cfg.trainer.epochs = 1  # Only 1 epoch for ultra-quick testing
        cfg.data.batch_size = 1  # Match data batch size
        cfg.trainer.accelerator = "cpu"  # Force CPU to avoid MPS issues
        cfg.trainer.device = "cpu"  # Force CPU device
        
        # Enable LoRA for quick training
        cfg.vision_encoder_cfg.lora = True
        cfg.vision_encoder_cfg.lora_r = 2  # Very small r value for speed
        
        # Set up logging for this experiment with LoRA suffix
        lora_suffix = "_lora" if cfg.vision_encoder_cfg.lora else "_wo_lora"
        exp_name = f"standard_temporal_{name}_win{temporal_window}_ctx{caption_context}{lora_suffix}"
        cfg.trainer.exp_name = exp_name
        cfg.trainer.output_dir = f"./outputs/{datetime.now().strftime('%Y-%m-%d')}/{exp_name}"
        
        # Initialize data module
        data_module = VideoDataModule(cfg)
        
        # Initialize model
        model = TrainerModule(cfg)
        
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
        
        # Force CPU usage to avoid MPS device issues
        accelerator = "cpu"
        devices = 1
        precision = "32"  # Use full precision to avoid MPS issues
        
        trainer = L.Trainer(
            max_epochs=cfg.trainer.epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            logger=logger,
            callbacks=[checkpoint_callback],
            gradient_clip_val=cfg.trainer.gradient_clip_val,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            enable_progress_bar=True
        )
        
        # Quick LoRA training with different temporal windows
        try:
            print(f"🚀 Quick LoRA training with temporal window {temporal_window}...")
            
            # Train for 1 epoch with LoRA
            trainer.fit(model, data_module)
            test_results = trainer.test(model, data_module, verbose=True)
            
            print(f"✅ LoRA training completed for temporal window {temporal_window}")
            print(f"📊 Final test loss: {test_results[0]['test/loss']:.4f}")
            
            # Store results
            self.results[name] = {
                'temporal_window': temporal_window,
                'caption_context': caption_context,
                'time_span_seconds': temporal_window / 30.0,
                'test_results': test_results,
                'config': dict(cfg)
            }
            
            print(f"✅ Completed evaluation for {name} configuration")
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            self.results[name] = {
                'temporal_window': temporal_window,
                'caption_context': caption_context,
                'time_span_seconds': temporal_window / 30.0,
                'error': str(e)
            }
    
    def run_standard_evaluation(self):
        """Run quick LoRA training on different temporal windows"""
        print("🎯 Quick LoRA Temporal Window Testing")
        print("="*60)
        print("Testing multiple temporal window configurations with LoRA:")
        print("- Ultra-short: 3, 5, 8 frames")
        print("- Short/Medium: 10, 12, 16, 24, 32 frames")
        print("- Long: 48, 64 frames (use on higher-memory machines)")
        print("Settings: batch_size=1, epochs=1, LoRA r=2")
        print("="*60)
        
        for config in self.standard_configs:
            self.evaluate_standard_config(config)
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save evaluation results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        lora_suffix = "_lora" if self.base_cfg.vision_encoder_cfg.lora else "_wo_lora"
        results_file = f"./outputs/standard_temporal_evaluation{lora_suffix}_{timestamp}.json"
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📊 Results saved to: {results_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("QUICK LORA TEMPORAL WINDOW TESTING SUMMARY")
        print("="*80)
        
        for config_name, result in self.results.items():
            if 'error' in result:
                print(f"❌ {config_name}: ERROR - {result['error']}")
            else:
                test_loss = result['test_results'][0]['test/loss'] if result['test_results'] else 'N/A'
                time_span = result['time_span_seconds']
                print(f"✅ {config_name}: Test Loss = {test_loss:.4f} (Time span: {time_span:.2f}s)")

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    """Main evaluation function for industry-standard temporal windows"""
    
    # Initialize evaluator
    evaluator = StandardTemporalEvaluator(cfg)
    
    # Run evaluation
    evaluator.run_standard_evaluation()

if __name__ == "__main__":
    main()

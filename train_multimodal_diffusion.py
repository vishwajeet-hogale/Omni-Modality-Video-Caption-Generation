import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from models.model import SwinBart
from models.diffusion_text_generator import DiffusionTextGenerator
from models.masked_language_model import MaskedLanguageModel
from data.datamodule import VideoDataModule
import hydra
from omegaconf import DictConfig

class MultimodalDiffusionTrainer(L.LightningModule):
    """Lightning module for training multimodal diffusion model"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        # Initialize model
        self.model = SwinBart(cfg)
        
        # Initialize loss functions
        self.diffusion_loss = nn.CrossEntropyLoss()
        self.masked_lm_loss = nn.CrossEntropyLoss()
        self.standard_loss = nn.CrossEntropyLoss()
        
        # Initialize metrics
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, images, audio, captions=None, is_new_video=False):
        """Forward pass with optional diffusion generation"""
        use_multimodal_diffusion = getattr(self.cfg, 'use_multimodal_diffusion', False)
        if use_multimodal_diffusion:
            return self.model.forward_with_diffusion(images, audio, captions, is_new_video)
        else:
            return self.model(images, captions, audio, is_new_video)
    
    def training_step(self, batch, batch_idx):
        """Training step with multiple loss components"""
        images, captions, audio = batch["images"], batch["captions"], batch["audio"]
        
        # Detect if this is a new video sequence
        is_new_video = (batch_idx == 0 or batch_idx != getattr(self, 'last_batch_idx', -1) + 1)
        self.last_batch_idx = batch_idx
        
        # Standard forward pass
        use_multimodal_diffusion = getattr(self.cfg, 'use_multimodal_diffusion', False)
        if use_multimodal_diffusion:
            loss, generated_captions = self.forward(images, audio, captions, is_new_video)
        else:
            outputs = self.forward(images, audio, captions, is_new_video)
            loss = outputs.loss
        
        # Add diffusion loss if enabled
        diffusion_loss = 0.0
        if use_multimodal_diffusion:
            try:
                # Generate with diffusion
                diffusion_captions = self.model.generate_with_diffusion(images, audio)
                diffusion_loss = self.diffusion_loss(
                    diffusion_captions.view(-1, diffusion_captions.size(-1)),
                    captions.view(-1)
                )
            except Exception as e:
                print(f"Warning: Diffusion generation failed: {e}")
                diffusion_loss = 0.0
        
        # Add masked LM loss if enabled
        masked_lm_loss = 0.0
        if use_multimodal_diffusion:
            try:
                # Generate with masking
                predictions, mask_positions = self.model.generate_with_masking(
                    images, audio, captions
                )
                masked_lm_loss = self.masked_lm_loss(
                    predictions, 
                    captions[mask_positions]
                )
            except Exception as e:
                print(f"Warning: Masked LM generation failed: {e}")
                masked_lm_loss = 0.0
        
        # Total loss
        total_loss = loss + 0.1 * diffusion_loss + 0.1 * masked_lm_loss
        
        # Log losses
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/standard_loss', loss, on_step=True, on_epoch=True)
        self.log('train/diffusion_loss', diffusion_loss, on_step=True, on_epoch=True)
        self.log('train/masked_lm_loss', masked_lm_loss, on_step=True, on_epoch=True)
        
        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log('train/lr', lr, on_step=True, on_epoch=False)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images, captions, audio = batch["images"], batch["captions"], batch["audio"]
        
        # Detect if this is a new video sequence
        is_new_video = (batch_idx == 0 or batch_idx != getattr(self, '_val_last_batch_idx', -1) + 1)
        self._val_last_batch_idx = batch_idx
        
        # Forward pass
        use_multimodal_diffusion = getattr(self.cfg, 'use_multimodal_diffusion', False)
        if use_multimodal_diffusion:
            loss, generated_captions = self.forward(images, audio, captions, is_new_video)
        else:
            outputs = self.forward(images, audio, captions, is_new_video)
            loss = outputs.loss
        
        # Log validation loss
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_start(self):
        """Initialize validation batch tracking"""
        self._val_last_batch_idx = -1
    
    def test_step(self, batch, batch_idx):
        """Test step mirrors validation logic for Lightning's Trainer.test()."""
        images, captions, audio = batch["images"], batch["captions"], batch["audio"]
        
        # Detect if this is a new video sequence in test loop
        is_new_video = (batch_idx == 0 or batch_idx != getattr(self, '_test_last_batch_idx', -1) + 1)
        self._test_last_batch_idx = batch_idx
        
        use_multimodal_diffusion = getattr(self.cfg, 'use_multimodal_diffusion', False)
        if use_multimodal_diffusion:
            loss, _ = self.forward(images, audio, captions, is_new_video)
        else:
            outputs = self.forward(images, audio, captions, is_new_video)
            loss = outputs.loss
        
        # Log test loss
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_test_epoch_start(self):
        """Initialize test batch tracking"""
        self._test_last_batch_idx = -1
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.cfg.trainer.lr,
            weight_decay=self.cfg.trainer.weight_decay
        )
        
        if self.cfg.trainer.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.cfg.trainer.epochs
            )
            return [optimizer], [scheduler]
        elif self.cfg.trainer.lr_scheduler == "constant":
            return optimizer
        else:
            return optimizer
    
    def on_train_epoch_start(self):
        """Initialize training batch tracking"""
        self.last_batch_idx = -1

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    """Main training function"""
    
    # Set up logging
    logger = TensorBoardLogger(
        save_dir=cfg.trainer.output_dir,
        name=cfg.trainer.exp_name,
        version=None
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.trainer.output_dir,
        filename='best_model',
        save_top_k=1,
        monitor='val/loss',
        mode='min',
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val/loss',
        patience=5,
        mode='min'
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps
    )
    
    # Initialize data module
    data_module = VideoDataModule(cfg)
    
    # Initialize model
    model = MultimodalDiffusionTrainer(cfg)
    
    # Print model info
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)

if __name__ == "__main__":
    main()

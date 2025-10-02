import os
import torch
import lightning as L
from models.model import SwinBart
import torchvision

class TrainerModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = SwinBart(cfg)
        self.best_val_loss = float("inf")
        self.save_dir = cfg.trainer.output_dir
        self.save_path = os.path.join(self.save_dir, "best_model.pth")
        os.makedirs(self.save_dir, exist_ok=True)
        # Save hyperparameters to logger
        self.save_hyperparameters(cfg)
        
        # Track video boundaries for temporal memory management
        self.current_video_name = None
        self.last_batch_idx = -1

    def forward(self, images, captions, audio, is_new_video=False):
        return self.model(images, captions, audio, is_new_video=is_new_video)

    def training_step(self, batch, batch_idx):
        images, captions, audio = batch["images"], batch["captions"], batch["audio"]
        
        # Detect if this is a new video sequence
        # This happens if batch_idx is not consecutive (indicating video boundary crossing)
        is_new_video = (batch_idx == 0 or batch_idx != self.last_batch_idx + 1)
        self.last_batch_idx = batch_idx
        
        # Log batch information
        batch_size = images.shape[0]
        image_shape = images.shape
        audio_shape = audio.shape
        num_captions = len(captions)
        
        # Log batch metrics
        self.log("train/batch_size", batch_size, on_step=True, logger=True)
        self.log("train/num_captions", num_captions, on_step=True, logger=True)
        
        # Print batch info for first few batches
        if batch_idx < 3:
            # Calculate memory usage
            image_memory = images.numel() * images.element_size() / (1024**2)  # MB
            audio_memory = audio.numel() * audio.element_size() / (1024**2)  # MB
            total_memory = image_memory + audio_memory
            
            print(f"Train Batch {batch_idx}: size={batch_size}, images={image_shape}, audio={audio_shape}, captions={num_captions}")
            print(f"Memory: images={image_memory:.2f}MB, audio={audio_memory:.2f}MB, total={total_memory:.2f}MB")
            if is_new_video and getattr(self.model, 'use_temporal_attention', False):
                print(f"  -> New video sequence detected, temporal memory will be reset")
        
        # Get previous captions for text feedback (use ground truth during training)
        previous_captions = None
        if hasattr(self, '_previous_captions') and not is_new_video:
            previous_captions = self._previous_captions
        
        # Forward pass with new video indicator and previous captions
        outputs = self(images, captions, audio, is_new_video=is_new_video, previous_captions=previous_captions)
        loss = outputs.loss
        
        # Store current captions for next iteration (use ground truth for stability)
        self._previous_captions = captions
        if is_new_video:
            self._previous_captions = None

        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Optional: log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, logger=True)
        
        if batch_idx == 0 and isinstance(self.logger, L.pytorch.loggers.TensorBoardLogger):
            # Log first 4 images
            grid = images[:4]
            self.logger.experiment.add_images("train/images", grid, self.current_epoch)

            # Log MFCC as image (assuming shape [B, C, H, W] or [B, H, W])
            mfccs = audio[:4]  # Shape: [B, C, H, W] or [B, H, W]
            if mfccs.dim() == 3:  # If [B, H, W], add channel dimension
                mfccs = mfccs.unsqueeze(1)
            mfcc_grid = torchvision.utils.make_grid(mfccs, normalize=True, scale_each=True)
            self.logger.experiment.add_image("train/audio_mfcc", mfcc_grid, self.current_epoch)
            
        return loss

    def validation_step(self, batch, batch_idx):
        images, captions, audio = batch["images"], batch["captions"], batch["audio"]
        
        # Detect if this is a new video sequence for validation
        is_new_video = (batch_idx == 0)  # For validation, reset at the start of each epoch
        
        # Log batch information
        batch_size = images.shape[0]
        image_shape = images.shape
        audio_shape = audio.shape
        num_captions = len(captions)
        
        # Log batch metrics
        self.log("val/batch_size", batch_size, on_step=True, logger=True)
        self.log("val/num_captions", num_captions, on_step=True, logger=True)
        
        # Print batch info for first few batches
        if batch_idx < 3:
            # Calculate memory usage
            image_memory = images.numel() * images.element_size() / (1024**2)  # MB
            audio_memory = audio.numel() * audio.element_size() / (1024**2)  # MB
            total_memory = image_memory + audio_memory
            
            print(f"Val Batch {batch_idx}: size={batch_size}, images={image_shape}, audio={audio_shape}, captions={num_captions}")
            print(f"Memory: images={image_memory:.2f}MB, audio={audio_memory:.2f}MB, total={total_memory:.2f}MB")
            if is_new_video and getattr(self.model, 'use_temporal_attention', False):
                print(f"  -> Validation started, temporal memory reset")
        
        # Get previous captions for validation (use ground truth during validation too)
        previous_captions = None
        if hasattr(self, '_val_previous_captions') and not is_new_video:
            previous_captions = self._val_previous_captions
        
        # Forward pass with new video indicator and previous captions
        outputs = self(images, captions, audio, is_new_video=is_new_video, previous_captions=previous_captions)
        loss = outputs.loss
        
        # Store current captions for next validation iteration
        self._val_previous_captions = captions
        if is_new_video:
            self._val_previous_captions = None

        self.log("val/loss_step", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("val/loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log sample inputs to TensorBoard
        if batch_idx == 0 and isinstance(self.logger, L.pytorch.loggers.TensorBoardLogger):
            # Log first 4 images
            grid = images[:4]
            self.logger.experiment.add_images("val/images", grid, self.current_epoch)

            # Log MFCC as image (assuming shape [B, C, H, W] or [B, H, W])
            mfccs = audio[:4]  # Shape: [B, C, H, W] or [B, H, W]
            if mfccs.dim() == 3:  # If [B, H, W], add channel dimension
                mfccs = mfccs.unsqueeze(1)
            mfcc_grid = torchvision.utils.make_grid(mfccs, normalize=True, scale_each=True)
            self.logger.experiment.add_image("val/audio_mfcc", mfcc_grid, self.current_epoch)

            # Optional: log raw audio (if you have it and want to hear it)
            # self.logger.experiment.add_audio("val/audio_waveform", raw_audio[0], self.current_epoch, sample_rate=16000)
            
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.trainer.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_epoch_start(self):
        """Log dataloader information at the start of each training epoch."""
        train_loader = self.trainer.train_dataloader
        val_loader = self.trainer.val_dataloaders if self.trainer.val_dataloaders else None
        
        print(f"\n📊 Epoch {self.current_epoch} DataLoader Info:")
        print(f"  Train batches: {len(train_loader)}")
        if val_loader:
            print(f"  Val batches: {len(val_loader)}")
        
        # Log total samples
        train_samples = len(train_loader.dataset)
        val_samples = len(val_loader.dataset) if val_loader else 0
        print(f"  Train samples: {train_samples}")
        print(f"  Val samples: {val_samples}")
        
        # Log batch size
        batch_size = train_loader.batch_size
        print(f"  Batch size: {batch_size}")
        print(f"  Expected train steps: {len(train_loader)}")
        if val_loader:
            print(f"  Expected val steps: {len(val_loader)}")
        
        # Reset temporal memory at epoch start
        if getattr(self.model, 'use_temporal_attention', False):
            self.model.reset_temporal_memory()
            print(f"  🔄 Temporal memory reset for new epoch")
        
        # Reset batch tracking
        self.last_batch_idx = -1

    def on_train_epoch_end(self):
        """Log epoch-level statistics."""
        # Get epoch metrics
        train_loss = self.trainer.callback_metrics.get("train/loss_epoch", 0)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        
        print(f"\n📈 Epoch {self.current_epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Learning Rate: {lr:.6f}")
        print(f"  Global Step: {self.global_step}")

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val/loss_epoch")
        if val_loss is None:
            return

        try:
            current = float(val_loss.detach().cpu().item())
        except Exception:
            current = float(val_loss)

        if current < self.best_val_loss and self.trainer.is_global_zero:
            self.best_val_loss = current
            self._save_best_model(current)

    def _save_best_model(self, current_val_loss: float):
        torch.save(self.model.state_dict(), self.save_path)
        archive_path = os.path.join(
            self.save_dir,
            f"best_step{self.global_step}_epoch{self.current_epoch}_loss{current_val_loss:.4f}.pth"
        )

        # Save archive as well
        # torch.save(self.model.state_dict(), archive_path)

        print(
            f"✅ New best model (val_loss={current_val_loss:.4f}) saved to:\n"
            f"  - {self.save_path} (latest best)\n"
            f"  - {archive_path} (archive)"
        )

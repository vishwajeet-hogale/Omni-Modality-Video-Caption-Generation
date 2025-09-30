# datamodule_basic.py
import lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from data.dataset import VideoCaptionDatasetCSV, VideoCaptionDataset
from data.split_utils import get_video_split, get_split_for_stage
import os


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.splits = None

    def setup(self, stage: str = None):
        d = self.cfg.data
        
        # Create video splits to prevent temporal leakage
        split_file = os.path.join(d.captions_dir, "video_splits.json")
        self.splits = get_video_split(
            captions_dir=d.captions_dir,
            train_ratio=getattr(d, 'train_ratio', 0.7),
            val_ratio=getattr(d, 'val_ratio', 0.15), 
            test_ratio=getattr(d, 'test_ratio', 0.15),
            random_seed=getattr(d, 'random_seed', 42),
            split_file=split_file
        )
        
        if self.cfg.data.use_csv:
            # Create separate datasets for each split
            train_videos = get_split_for_stage(self.splits, 'train')
            val_videos = get_split_for_stage(self.splits, 'val')
            test_videos = get_split_for_stage(self.splits, 'test')
            
            self.train_dataset = VideoCaptionDatasetCSV(
                captions_dir=d.captions_dir,
                frames_dir=d.frames_dir,
                audio_dir=d.audio_dir,
                split='train',
                video_names=train_videos
            )
            
            self.val_dataset = VideoCaptionDatasetCSV(
                captions_dir=d.captions_dir,
                frames_dir=d.frames_dir,
                audio_dir=d.audio_dir,
                split='val',
                video_names=val_videos
            )
            
            self.test_dataset = VideoCaptionDatasetCSV(
                captions_dir=d.captions_dir,
                frames_dir=d.frames_dir,
                audio_dir=d.audio_dir,
                split='test',
                video_names=test_videos
            )
        else:
            # For non-CSV datasets, use the same split logic
            train_videos = get_split_for_stage(self.splits, 'train')
            val_videos = get_split_for_stage(self.splits, 'val')
            test_videos = get_split_for_stage(self.splits, 'test')
            
            self.train_dataset = VideoCaptionDataset(
                frames_dir=d.frames_dir,
                captions_dir=d.captions_dir,
                split='train',
                video_names=train_videos
            )
            
            self.val_dataset = VideoCaptionDataset(
                frames_dir=d.frames_dir,
                captions_dir=d.captions_dir,
                split='val',
                video_names=val_videos
            )
            
            self.test_dataset = VideoCaptionDataset(
                frames_dir=d.frames_dir,
                captions_dir=d.captions_dir,
                split='test',
                video_names=test_videos
            )

    def train_dataloader(self):
        d = self.cfg.data
        return DataLoader(
            self.train_dataset,
            batch_size=d.batch_size,
            num_workers=d.num_workers,
            pin_memory=d.pin_memory,
            collate_fn=VideoCaptionDatasetCSV.collate_fn if self.cfg.data.use_csv else None,
            shuffle=True,
        )

    def val_dataloader(self):
        d = self.cfg.data
        return DataLoader(
            self.val_dataset,
            batch_size=d.batch_size,
            num_workers=d.num_workers,
            pin_memory=d.pin_memory,
            collate_fn=VideoCaptionDatasetCSV.collate_fn if self.cfg.data.use_csv else None,
            shuffle=False,  # Don't shuffle validation data
        )

    def test_dataloader(self):
        d = self.cfg.data
        return DataLoader(
            self.test_dataset,
            batch_size=d.batch_size,
            num_workers=d.num_workers,
            pin_memory=d.pin_memory,
            collate_fn=VideoCaptionDatasetCSV.collate_fn if self.cfg.data.use_csv else None,
            shuffle=False,
        )

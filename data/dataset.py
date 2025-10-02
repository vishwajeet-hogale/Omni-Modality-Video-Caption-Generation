import os
import json
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Sampler
import random

class VideoSequentialSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        print(f"VideoSequentialSampler initialized:")
        print(f"Batch size: {batch_size}")
        print(f"Total videos: {len(dataset.video_boundaries)}")
        
    def __iter__(self):
        # Create batches that stay within video boundaries
        print(f"\nCreating sequential batches...")
        indices = []
        batch_count = 0
        # Process videos in order and separately
        for start_idx, end_idx, video_name in self.dataset.video_boundaries:
            video_frames = list(range(start_idx, end_idx))
            print(f"Video '{video_name}' frames: {len(video_frames)}")
            # Create batches within this video
            for i in range(0, len(video_frames), self.batch_size):
                batch = video_frames[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Only full batches
                    indices.extend(batch)
        
        return iter(indices)
    
    def __len__(self):
        # Calculate total number of samples
        total = 0
        for start_idx, end_idx, _ in self.dataset.video_boundaries:
            video_frames = end_idx - start_idx
            total += (video_frames // self.batch_size) * self.batch_size
        return total

class VideoCaptionDataset(Dataset):
    def __init__(self, captions_dir, frames_dir, transform=None, split=None, video_names=None):
        """
        Args:
            captions_dir (str): Path to the directory with JSON caption files.
            frames_dir (str): Path to the directory with frame subdirectories.
            transform (callable, optional): Optional transform to be applied on an image.
            split (str): Split name for logging purposes.
            video_names (list): List of video names to include in this dataset.
        """
        self.captions_dir = captions_dir
        self.frames_dir = frames_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.split = split
        self.video_names = video_names

        self.data = [] 

        # Get all available video files
        all_video_files = [f for f in os.listdir(captions_dir) if f.endswith(".json")]
        
        # If video_names is specified, filter to only those videos
        if video_names is not None:
            video_files = [f for f in all_video_files if os.path.splitext(f)[0] in video_names]
        else:
            video_files = all_video_files

        for json_file in video_files:
            video_name = os.path.splitext(json_file)[0]
            json_path = os.path.join(captions_dir, json_file)

            with open(json_path, "r") as f:
                frame_caption_map = json.load(f)

            for frame_id, caption in frame_caption_map.items():
                image_path = os.path.join(frames_dir, video_name, f"{frame_id}")
                if os.path.exists(image_path):
                    self.data.append((image_path, caption))
                    # current_idx += 1
                else:
                    print(f"Warning: Missing image {image_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, caption = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, caption


class VideoCaptionDatasetCSV(Dataset):
    def __init__(self, captions_dir, frames_dir, audio_dir, transform=None, split=None, video_names=None):
        self.captions_dir = captions_dir
        self.frames_dir = frames_dir
        self.audio_dir = audio_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.split = split
        self.video_names = video_names

        self.data = [] 
        self.video_boundaries = []  # Track where each video starts/ends
        current_idx = 0
        # Get all available video files
        all_video_files = [f for f in os.listdir(captions_dir) if f.endswith(".csv")]
        
        # If video_names is specified, filter to only those videos
        if video_names is not None:
            video_files = [f for f in all_video_files if os.path.splitext(f)[0] in video_names]
        else:
            video_files = all_video_files

        for csv_file in video_files:
            video_name = os.path.splitext(csv_file)[0]
            csv_path = os.path.join(captions_dir, csv_file)
            df = pd.read_csv(csv_path)

            start_idx = current_idx
            print(f"Loading video: '{video_name}' - {len(df)} frames")

            for frame_num, row in df.iterrows():
                frame_id = f"frame_{str(frame_num).zfill(4)}.jpg"
                caption = row['caption']
                image_path = os.path.join(frames_dir, video_name, frame_id)

                audio_path = os.path.join(audio_dir, video_name, f"frame_{str(frame_num).zfill(4)}.npy")

                if not os.path.exists(audio_path):
                    print(f"Warning: Missing audio {audio_path}")
                if os.path.exists(image_path):
                    self.data.append((image_path, audio_path, caption))
                    current_idx += 1 
                else:
                    print(f"Warning: Missing image {image_path}")
            # Record video boundaries
            self.video_boundaries.append((start_idx, current_idx, video_name))
            print(f"Video '{video_name}' boundaries: {start_idx} to {current_idx-1} ({current_idx-start_idx} frames)")
        print(f"\nDataset Summary:")
        print(f"   Total videos: {len(self.video_boundaries)}")
        print(f"   Total frames: {len(self.data)}")
        for i, (start, end, name) in enumerate(self.video_boundaries):
            print(f"   Video {i+1}: '{name}' - {end-start} frames")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, audio_path, caption = self.data[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # audio
        audio_tensor = torch.tensor(np.load(audio_path), dtype=torch.float32)  # (n_mfcc, T)
        return {"images": image, "captions": caption, "audio": audio_tensor}

    @staticmethod
    def collate_fn(batch):
        images = [b["images"] for b in batch]
        captions = [b["captions"] for b in batch]
        audios = [b["audio"] for b in batch]

        # Stack images
        images_tensor = torch.stack(images)

        # Pad audio sequences
        max_T = max(a.shape[1] for a in audios)  # find longest time dimension
        n_mfcc = audios[0].shape[0]
        B = len(audios)
        audio_tensor = torch.zeros(B, n_mfcc, max_T, dtype=audios[0].dtype)

        audio_lengths = []
        for i, a in enumerate(audios):
            T = a.shape[1]
            audio_tensor[i, :, :T] = a
            audio_lengths.append(T)
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

        return {
            "images": images_tensor,
            "captions": captions,
            "audio": audio_tensor,         # (B, n_mfcc, T_max)
            "audio_lengths": audio_lengths # (B,) actual lengths
        }
        
if __name__ == "__main__":
    captions_dir= "/Users/rohitkulkarni/Desktop/Omni-Modality-Video-Caption-Generation/captions_csv"
    frames_dir= "/Users/rohitkulkarni/Desktop/Omni-Modality-Video-Caption-Generation/frames"
    audio_dir= "/Users/rohitkulkarni/Desktop/Omni-Modality-Video-Caption-Generation/mfccs"
    dataset = VideoCaptionDatasetCSV(captions_dir=captions_dir, frames_dir=frames_dir, audio_dir=audio_dir)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=VideoCaptionDatasetCSV.collate_fn)
    for batch in dataloader:
        print(batch["images"].shape)  # Should print (B, 3, H, W)
        print(batch["captions"])       # Should print list of captions
        print(batch["audio"].shape)    # Should print (B, n_mfcc, T_max)
        print(batch["audio_lengths"])   # Should print (B,) actual lengths
        break
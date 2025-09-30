import os
import random
from typing import List, Tuple, Dict
import json


def get_video_split(captions_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, 
                   random_seed: int = 42, split_file: str = None) -> Dict[str, List[str]]:
    """
    Create video-level train/val/test splits to prevent temporal leakage.
    
    Args:
        captions_dir: Directory containing CSV caption files
        train_ratio: Proportion of videos for training
        val_ratio: Proportion of videos for validation  
        test_ratio: Proportion of videos for testing
        random_seed: Random seed for reproducible splits
        split_file: Optional path to save/load split configuration
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing lists of video names
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Get all video files
    video_files = [f for f in os.listdir(captions_dir) if f.endswith(".csv")]
    video_names = [os.path.splitext(f)[0] for f in video_files]
    
    # Sort video names for consistent ordering
    video_names.sort()
    
    # Check if split file exists and load it
    if split_file and os.path.exists(split_file):
        with open(split_file, 'r') as f:
            splits = json.load(f)
        print(f"Loaded existing split from {split_file}")
        return splits
    
    # Create new split
    n_videos = len(video_names)
    n_train = max(1, int(n_videos * train_ratio))
    n_val = max(1, int(n_videos * val_ratio))
    n_test = n_videos - n_train - n_val
    
    # Ensure we have at least one video in each split
    if n_test < 1:
        n_test = 1
        n_train = max(1, n_train - 1)
    if n_val < 1:
        n_val = 1
        n_train = max(1, n_train - 1)
    
    # Shuffle video names
    shuffled_videos = video_names.copy()
    random.shuffle(shuffled_videos)
    
    # Create splits
    train_videos = shuffled_videos[:n_train]
    val_videos = shuffled_videos[n_train:n_train + n_val]
    test_videos = shuffled_videos[n_train + n_val:]
    
    splits = {
        'train': train_videos,
        'val': val_videos, 
        'test': test_videos
    }
    
    # Save split configuration
    if split_file:
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        with open(split_file, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"Saved split configuration to {split_file}")
    
    # Print split information
    print(f"Video Split Summary:")
    print(f"  Total videos: {n_videos}")
    print(f"  Train: {len(train_videos)} videos - {train_videos}")
    print(f"  Val: {len(val_videos)} videos - {val_videos}")
    print(f"  Test: {len(test_videos)} videos - {test_videos}")
    
    return splits


def get_split_for_stage(splits: Dict[str, List[str]], stage: str) -> List[str]:
    """
    Get video names for a specific stage (train/val/test).
    
    Args:
        splits: Dictionary containing train/val/test video lists
        stage: Stage name ('train', 'val', or 'test')
        
    Returns:
        List of video names for the specified stage
    """
    if stage not in splits:
        raise ValueError(f"Invalid stage: {stage}. Must be one of {list(splits.keys())}")
    
    return splits[stage]

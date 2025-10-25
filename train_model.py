"""
Main training script for deepfake detection model

Usage:
    python train_model.py

Make sure to organize your dataset in data/train and data/val folders first.
"""

import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from src.data_processing.video_processor import VideoProcessor
from src.data_processing.dataset import DeepfakeDataset
from src.models.deepfake_detector import DeepfakeDetector
from src.training.train import Trainer

def load_dataset_paths(data_dir, split='train'):
    """
    Load video paths and labels from organized directory structure
    
    Expected structure:
        data/
        ├── train/
        │   ├── real/
        │   │   ├── video1.mp4
        │   │   └── video2.mp4
        │   └── fake/
        │       ├── video1.mp4
        │       └── video2.mp4
        └── val/
            ├── real/
            └── fake/
    """
    base_path = Path(data_dir) / split
    
    video_paths = []
    labels = []
    
    # Load real videos (label 0)
    real_path = base_path / 'real'
    if real_path.exists():
        for video_file in real_path.glob('*.mp4'):
            video_paths.append(str(video_file))
            labels.append(0)
        for video_file in real_path.glob('*.avi'):
            video_paths.append(str(video_file))
            labels.append(0)
        for video_file in real_path.glob('*.mov'):
            video_paths.append(str(video_file))
            labels.append(0)
    
    # Load fake videos (label 1)
    fake_path = base_path / 'fake'
    if fake_path.exists():
        for video_file in fake_path.glob('*.mp4'):
            video_paths.append(str(video_file))
            labels.append(1)
        for video_file in fake_path.glob('*.avi'):
            video_paths.append(str(video_file))
            labels.append(1)
        for video_file in fake_path.glob('*.mov'):
            video_paths.append(str(video_file))
            labels.append(1)
    
    return video_paths, labels

def main():
    # Load configuration
    print("Loading configuration...")
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    print("Loading dataset paths...")
    train_videos, train_labels = load_dataset_paths('data', split='train')
    val_videos, val_labels = load_dataset_paths('data', split='val')
    
    print(f"Training videos: {len(train_videos)} ({train_labels.count(0)} real, {train_labels.count(1)} fake)")
    print(f"Validation videos: {len(val_videos)} ({val_labels.count(0)} real, {val_labels.count(1)} fake)")
    
    if len(train_videos) == 0:
        print("\n⚠️  No training videos found!")
        print("Please organize your dataset in the following structure:")
        print("  data/")
        print("  ├── train/")
        print("  │   ├── real/  # Put real videos here")
        print("  │   └── fake/  # Put fake videos here")
        print("  └── val/")
        print("      ├── real/")
        print("      └── fake/")
        return
    
    # Create datasets
    print("\nInitializing video processor...")
    processor = VideoProcessor()
    
    print("Creating datasets...")
    train_dataset = DeepfakeDataset(
        train_videos, 
        train_labels, 
        processor,
        num_frames=config['data']['num_frames']
    )
    val_dataset = DeepfakeDataset(
        val_videos, 
        val_labels, 
        processor,
        num_frames=config['data']['num_frames']
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility, increase if you have issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\nInitializing model...")
    model = DeepfakeDetector(num_classes=config['model']['num_classes'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Start training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.train(config['training']['epochs'])
    
    print("\n" + "="*60)
    print("Training complete!")
    print("Best model saved to: models/checkpoints/best_model.pth")
    print("="*60)

if __name__ == "__main__":
    main()

"""
Helper script to prepare and organize your dataset

This script helps you organize videos into the correct directory structure
and provides dataset statistics.
"""

from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

def create_directory_structure():
    """Create the required directory structure for training"""
    directories = [
        "data/train/real",
        "data/train/fake",
        "data/val/real",
        "data/val/fake",
        "models/checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\n✅ Directory structure created!")
    print("\nNow place your videos in the following folders:")
    print("  - Training real videos: data/train/real/")
    print("  - Training fake videos: data/train/fake/")
    print("  - Validation real videos: data/val/real/")
    print("  - Validation fake videos: data/val/fake/")

def organize_dataset(raw_data_path, output_path):
    """
    Organize videos into train/val/test splits
    Expected structure:
    raw_data_path/
        real/
            video1.mp4
            video2.mp4
        fake/
            video1.mp4
            video2.mp4
    """
    output = Path(output_path)
    output.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        (output / split / 'real').mkdir(parents=True, exist_ok=True)
        (output / split / 'fake').mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Organizing dataset from {raw_data_path} to {output_path}...")
    
    for label in ['real', 'fake']:
        source_path = Path(raw_data_path, label)
        if not source_path.exists():
            print(f"⚠️  Warning: {source_path} does not exist")
            continue
            
        videos = list(source_path.glob('*.mp4')) + \
                 list(source_path.glob('*.avi')) + \
                 list(source_path.glob('*.mov')) + \
                 list(source_path.glob('*.mkv'))
        
        if len(videos) == 0:
            print(f"⚠️  No videos found in {source_path}")
            continue
        
        # Split: 70% train, 15% val, 15% test
        train_vids, temp_vids = train_test_split(videos, test_size=0.3, random_state=42)
        val_vids, test_vids = train_test_split(temp_vids, test_size=0.5, random_state=42)
        
        # Copy files
        for vid in train_vids:
            shutil.copy(vid, output / 'train' / label / vid.name)
        for vid in val_vids:
            shutil.copy(vid, output / 'val' / label / vid.name)
        for vid in test_vids:
            shutil.copy(vid, output / 'test' / label / vid.name)
        
        print(f"✓ Processed {len(videos)} {label} videos")
    
    print(f"\n✅ Dataset organized in {output_path}")
    print(f"Train: {len(list((output/'train'/'real').glob('*')))} real, {len(list((output/'train'/'fake').glob('*')))} fake")
    print(f"Val: {len(list((output/'val'/'real').glob('*')))} real, {len(list((output/'val'/'fake').glob('*')))} fake")
    print(f"Test: {len(list((output/'test'/'real').glob('*')))} real, {len(list((output/'test'/'fake').glob('*')))} fake")

def count_videos(directory):
    """Count videos in a directory"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    count = 0
    for ext in video_extensions:
        count += len(list(Path(directory).glob(ext)))
    return count

def show_dataset_stats():
    """Display dataset statistics"""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    stats = {
        "Training - Real": count_videos("data/train/real"),
        "Training - Fake": count_videos("data/train/fake"),
        "Validation - Real": count_videos("data/val/real"),
        "Validation - Fake": count_videos("data/val/fake")
    }
    
    for category, count in stats.items():
        status = "✓" if count > 0 else "⚠️"
        print(f"{status} {category}: {count} videos")
    
    total_train = stats["Training - Real"] + stats["Training - Fake"]
    total_val = stats["Validation - Real"] + stats["Validation - Fake"]
    total = total_train + total_val
    
    print("-" * 60)
    print(f"Total Training: {total_train} videos")
    print(f"Total Validation: {total_val} videos")
    print(f"Total Dataset: {total} videos")
    print("=" * 60)
    
    if total == 0:
        print("\n⚠️  No videos found! Please add videos to the directories.")
    elif total < 100:
        print("\n⚠️  Warning: Dataset is very small. Consider adding more videos for better results.")
    else:
        print("\n✅ Dataset looks good! You can start training.")

def main():
    print("🎭 Deepfake Detection - Dataset Preparation Tool\n")
    
    choice = input("What would you like to do?\n"
                   "1. Create directory structure\n"
                   "2. Show dataset statistics\n"
                   "3. Organize raw dataset (auto-split train/val/test)\n"
                   "4. All of the above\n"
                   "Enter choice (1/2/3/4): ").strip()
    
    if choice in ['1', '4']:
        print("\n📁 Creating directory structure...")
        create_directory_structure()
    
    if choice in ['3', '4']:
        print("\n📊 Organizing raw dataset...")
        raw_path = input("Enter path to raw data folder (default: data/raw): ").strip() or "data/raw"
        output_path = input("Enter output path (default: data): ").strip() or "data"
        if Path(raw_path).exists():
            organize_dataset(raw_path, output_path)
        else:
            print(f"⚠️  Path {raw_path} does not exist. Please create it and add videos.")
    
    if choice in ['2', '4']:
        show_dataset_stats()
    
    print("\n" + "="*60)
    print("DATASET RECOMMENDATIONS")
    print("="*60)
    print("Minimum dataset size:")
    print("  - Training: 100+ videos (50 real, 50 fake)")
    print("  - Validation: 20+ videos (10 real, 10 fake)")
    print("\nRecommended dataset size:")
    print("  - Training: 1000+ videos (500 real, 500 fake)")
    print("  - Validation: 200+ videos (100 real, 100 fake)")
    print("\nSupported formats: MP4, AVI, MOV, MKV")
    print("="*60)

if __name__ == "__main__":
    main()

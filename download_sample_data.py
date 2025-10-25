"""
Sample data downloader and dataset information

This script provides information about public deepfake datasets
and helps you get started with training data.
"""

from pathlib import Path

def show_dataset_sources():
    """Display information about public deepfake datasets"""
    print("="*70)
    print("🎭 DEEPFAKE DETECTION DATASETS")
    print("="*70)
    print()
    
    datasets = {
        "FaceForensics++": {
            "url": "https://github.com/ondyari/FaceForensics",
            "description": "High-quality dataset with multiple manipulation methods",
            "size": "1000+ videos",
            "quality": "⭐⭐⭐⭐⭐",
            "difficulty": "Medium",
            "recommended": True
        },
        "Celeb-DF": {
            "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
            "description": "Celebrity deepfakes dataset",
            "size": "590 real + 5639 fake videos",
            "quality": "⭐⭐⭐⭐⭐",
            "difficulty": "Medium",
            "recommended": True
        },
        "DFDC (Kaggle)": {
            "url": "https://www.kaggle.com/c/deepfake-detection-challenge",
            "description": "Large-scale deepfake detection challenge dataset",
            "size": "100,000+ videos",
            "quality": "⭐⭐⭐⭐",
            "difficulty": "Hard (very large)",
            "recommended": False  # Too large for beginners
        },
        "DeeperForensics-1.0": {
            "url": "https://github.com/EndlessSora/DeeperForensics-1.0",
            "description": "Large-scale dataset with diverse perturbations",
            "size": "60,000 videos",
            "quality": "⭐⭐⭐⭐⭐",
            "difficulty": "Hard",
            "recommended": False
        },
        "FaceShifter": {
            "url": "https://github.com/ondyari/FaceForensics/tree/master/dataset",
            "description": "Part of FaceForensics++ with FaceShifter method",
            "size": "Included in FF++",
            "quality": "⭐⭐⭐⭐",
            "difficulty": "Medium",
            "recommended": True
        }
    }
    
    for i, (name, info) in enumerate(datasets.items(), 1):
        recommended = "⭐ RECOMMENDED" if info["recommended"] else ""
        print(f"{i}. {name} {recommended}")
        print(f"   URL: {info['url']}")
        print(f"   Description: {info['description']}")
        print(f"   Size: {info['size']}")
        print(f"   Quality: {info['quality']}")
        print(f"   Difficulty: {info['difficulty']}")
        print()
    
    print("="*70)

def show_download_instructions():
    """Show instructions for downloading datasets"""
    print("\n📥 HOW TO DOWNLOAD DATASETS")
    print("="*70)
    print()
    
    print("For FaceForensics++ (RECOMMENDED for beginners):")
    print("  1. Visit: https://github.com/ondyari/FaceForensics")
    print("  2. Fill out the access request form")
    print("  3. You'll receive download scripts via email")
    print("  4. Run the download script to get videos")
    print()
    
    print("For Celeb-DF:")
    print("  1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics")
    print("  2. Follow their download instructions")
    print("  3. Download the dataset (Google Drive link)")
    print()
    
    print("For DFDC (Kaggle):")
    print("  1. Install Kaggle CLI: pip install kaggle")
    print("  2. Set up Kaggle API credentials")
    print("  3. Run: kaggle competitions download -c deepfake-detection-challenge")
    print("  ⚠️  Warning: Very large dataset (470GB+)")
    print()
    
    print("="*70)

def show_data_organization():
    """Show how to organize downloaded data"""
    print("\n📁 DATA ORGANIZATION")
    print("="*70)
    print()
    print("After downloading, organize your data like this:")
    print()
    print("Option 1: Manual organization")
    print("  data/")
    print("  ├── train/")
    print("  │   ├── real/    # Put real videos here")
    print("  │   └── fake/    # Put fake videos here")
    print("  └── val/")
    print("      ├── real/")
    print("      └── fake/")
    print()
    
    print("Option 2: Auto-split from raw data")
    print("  data/")
    print("  └── raw/")
    print("      ├── real/    # Put all real videos here")
    print("      └── fake/    # Put all fake videos here")
    print()
    print("  Then run: python prepare_dataset.py")
    print("  Select option 3 to auto-split into train/val/test")
    print()
    print("="*70)

def create_sample_structure():
    """Create sample directory structure"""
    print("\n📂 Creating sample directory structure...")
    
    directories = [
        "data/raw/real",
        "data/raw/fake",
        "data/train/real",
        "data/train/fake",
        "data/val/real",
        "data/val/fake",
        "data/test/real",
        "data/test/fake"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created!")
    print("\nNow you can:")
    print("  1. Download datasets (see instructions above)")
    print("  2. Place videos in data/raw/real/ and data/raw/fake/")
    print("  3. Run: python prepare_dataset.py")
    print()

def show_quick_start():
    """Show quick start guide"""
    print("\n🚀 QUICK START GUIDE")
    print("="*70)
    print()
    print("Step 1: Create directories")
    print("  → Run this script with option 3")
    print()
    print("Step 2: Get a dataset")
    print("  → Download FaceForensics++ (recommended)")
    print("  → Or Celeb-DF for celebrity deepfakes")
    print()
    print("Step 3: Organize data")
    print("  → Place videos in data/raw/real/ and data/raw/fake/")
    print("  → Run: python prepare_dataset.py")
    print("  → Choose option 3 to auto-split")
    print()
    print("Step 4: Train model")
    print("  → Run: python train_model.py")
    print("  → Wait for training to complete (~2-4 hours)")
    print()
    print("Step 5: Use the system")
    print("  → Run: streamlit run app/streamlit_app.py")
    print("  → Upload videos and get results!")
    print()
    print("="*70)

def main():
    """Main function"""
    print("🎭 Deepfake Detection - Dataset Information Tool\n")
    
    while True:
        print("\nWhat would you like to do?")
        print("1. View available datasets")
        print("2. View download instructions")
        print("3. Create directory structure")
        print("4. View data organization guide")
        print("5. View quick start guide")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            show_dataset_sources()
        elif choice == '2':
            show_download_instructions()
        elif choice == '3':
            create_sample_structure()
        elif choice == '4':
            show_data_organization()
        elif choice == '5':
            show_quick_start()
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()

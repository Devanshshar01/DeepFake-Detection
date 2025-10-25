"""
Test script to verify installation and setup

This script checks:
1. All required packages are installed
2. PyTorch and device availability
3. Model can be instantiated
4. Video processor works
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('librosa', 'Librosa'),
        ('scipy', 'SciPy'),
        ('timm', 'timm'),
        ('sklearn', 'scikit-learn'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('streamlit', 'Streamlit'),
        ('plotly', 'Plotly'),
        ('tqdm', 'tqdm'),
        ('PIL', 'Pillow'),
        ('yaml', 'PyYAML'),
        ('albumentations', 'Albumentations')
    ]
    
    failed = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            failed.append(name)
    
    if failed:
        print(f"\n⚠️  Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_torch():
    """Test PyTorch installation and device availability"""
    print("\nTesting PyTorch...")
    import torch
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"  ✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f"  ✓ Using Apple Silicon GPU (MPS)")
    else:
        device = 'cpu'
        print(f"  ⚠️  Using CPU (training will be slow)")
    
    # Test tensor creation
    try:
        x = torch.randn(2, 3, 224, 224).to(device)
        print(f"  ✓ Created tensor on {device}: {x.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to create tensor: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe (optional but recommended)"""
    print("\nTesting MediaPipe...")
    try:
        import mediapipe as mp
        print(f"  ✓ MediaPipe version: {mp.__version__}")
        return True
    except ImportError:
        print("  ⚠️  MediaPipe not installed (face detection will be limited)")
        print("  Install with: pip install mediapipe")
        return False

def test_ffmpeg():
    """Test FFmpeg installation"""
    print("\nTesting FFmpeg...")
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"  ✓ {version_line}")
            return True
        else:
            print("  ✗ FFmpeg found but returned error")
            return False
    except FileNotFoundError:
        print("  ✗ FFmpeg not found")
        print("  Install with: brew install ffmpeg (macOS)")
        return False
    except Exception as e:
        print(f"  ✗ Error testing FFmpeg: {e}")
        return False

def test_model():
    """Test model instantiation"""
    print("\nTesting model...")
    try:
        from src.models.deepfake_detector import DeepfakeDetector
        import torch
        
        model = DeepfakeDetector()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        
        # Test forward pass
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        model = model.to(device)
        frames = torch.randn(1, 16, 3, 224, 224).to(device)
        audio = torch.randn(1, 13, 100).to(device)
        
        with torch.no_grad():
            output = model(frames, audio)
        
        print(f"  ✓ Forward pass successful: {output.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_processor():
    """Test video processor"""
    print("\nTesting video processor...")
    try:
        from src.data_processing.video_processor import VideoProcessor
        processor = VideoProcessor()
        print("  ✓ VideoProcessor initialized successfully")
        return True
    except Exception as e:
        print(f"  ✗ VideoProcessor test failed: {e}")
        return False

def test_config():
    """Test configuration file"""
    print("\nTesting configuration...")
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path('configs/config.yaml')
        if not config_path.exists():
            print("  ✗ Configuration file not found")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("  ✓ Configuration loaded successfully")
        print(f"    - Batch size: {config['data']['batch_size']}")
        print(f"    - Num frames: {config['data']['num_frames']}")
        print(f"    - Learning rate: {config['training']['learning_rate']}")
        return True
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False

def main():
    print("="*70)
    print("🎭 DEEPFAKE DETECTION - INSTALLATION TEST")
    print("="*70)
    
    tests = [
        ("Package Imports", test_imports),
        ("PyTorch", test_torch),
        ("MediaPipe", test_mediapipe),
        ("FFmpeg", test_ffmpeg),
        ("Configuration", test_config),
        ("Model", test_model),
        ("Video Processor", test_video_processor)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("-"*70)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("  1. Run: python prepare_dataset.py")
        print("  2. Add your training videos to data/train/")
        print("  3. Run: python train_model.py")
        print("  4. Launch web app: python -m streamlit run app/streamlit_app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    print("="*70)

if __name__ == "__main__":
    main()

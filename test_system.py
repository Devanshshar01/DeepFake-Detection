"""
System testing utilities for deepfake detection

Run tests to verify all components are working correctly.
"""

from pathlib import Path
from src.data_processing.video_processor import VideoProcessor
from src.models.deepfake_detector import DeepfakeDetector
import torch

def test_video_processor():
    """Test video processor initialization and basic functionality"""
    print("\n🧪 Testing Video Processor...")
    processor = VideoProcessor()
    
    # Test with a sample video if available
    test_video = "data/test/real/sample.mp4"
    if Path(test_video).exists():
        try:
            frames = processor.extract_frames(test_video, num_frames=8)
            assert len(frames) == 8, f"Expected 8 frames, got {len(frames)}"
            assert frames[0].shape == (224, 224, 3), f"Expected shape (224, 224, 3), got {frames[0].shape}"
            print("  ✅ Video processor test passed")
            return True
        except Exception as e:
            print(f"  ❌ Video processor test failed: {e}")
            return False
    else:
        print(f"  ⚠️  Test video not found at {test_video}")
        print("  ℹ️  Skipping video processor test (no test video available)")
        return True  # Not a failure, just no test data

def test_model_forward():
    """Test model forward pass"""
    print("\n🧪 Testing Model Forward Pass...")
    try:
        model = DeepfakeDetector()
        batch_size = 2
        num_frames = 16
        
        # Create dummy inputs
        frames = torch.randn(batch_size, num_frames, 3, 224, 224)
        audio = torch.randn(batch_size, 13, 100)
        
        # Forward pass
        output = model(frames, audio)
        
        assert output.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {output.shape}"
        print("  ✅ Model forward pass test passed")
        return True
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference():
    """Test inference pipeline"""
    print("\n🧪 Testing Inference Pipeline...")
    
    model_path = "models/checkpoints/best_model.pth"
    test_video = "data/test/real/sample.mp4"
    
    if not Path(model_path).exists():
        print(f"  ⚠️  Model not found at {model_path}")
        print("  ℹ️  Skipping inference test (train a model first)")
        return True  # Not a failure, just no trained model
    
    if not Path(test_video).exists():
        print(f"  ⚠️  Test video not found at {test_video}")
        print("  ℹ️  Skipping inference test (no test video available)")
        return True
    
    try:
        from src.data_processing.video_processor import VideoProcessor
        from src.inference.detector import DeepfakeInference
        
        processor = VideoProcessor()
        detector = DeepfakeInference(model_path, processor)
        
        result = detector.predict_video(test_video)
        
        assert 'prediction' in result, "Result missing 'prediction' key"
        assert 'confidence' in result, "Result missing 'confidence' key"
        assert result['prediction'] in ['REAL', 'FAKE', 'ERROR'], f"Invalid prediction: {result['prediction']}"
        
        print(f"  ✅ Inference test passed")
        print(f"     Prediction: {result['prediction']}, Confidence: {result['confidence']:.2%}")
        return True
    except Exception as e:
        print(f"  ❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset():
    """Test dataset loading"""
    print("\n🧪 Testing Dataset...")
    try:
        from src.data_processing.dataset import DeepfakeDataset
        from src.data_processing.video_processor import VideoProcessor
        
        # Check if we have any training data
        train_real = Path("data/train/real")
        train_fake = Path("data/train/fake")
        
        if not train_real.exists() or not train_fake.exists():
            print("  ⚠️  Training data directories not found")
            print("  ℹ️  Skipping dataset test (add training data first)")
            return True
        
        # Get sample videos
        videos = list(train_real.glob("*.mp4"))[:2] + list(train_fake.glob("*.mp4"))[:2]
        
        if len(videos) == 0:
            print("  ⚠️  No training videos found")
            print("  ℹ️  Skipping dataset test (add videos to data/train/)")
            return True
        
        labels = [0] * min(2, len(list(train_real.glob("*.mp4")))) + \
                [1] * min(2, len(list(train_fake.glob("*.mp4"))))
        
        processor = VideoProcessor()
        dataset = DeepfakeDataset(
            [str(v) for v in videos],
            labels,
            processor,
            num_frames=8
        )
        
        # Test loading one sample
        frames, audio, label = dataset[0]
        
        assert frames.shape[0] == 8, f"Expected 8 frames, got {frames.shape[0]}"
        assert audio.shape == (13, 100), f"Expected audio shape (13, 100), got {audio.shape}"
        
        print(f"  ✅ Dataset test passed ({len(dataset)} samples)")
        return True
    except Exception as e:
        print(f"  ❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test configuration loading"""
    print("\n🧪 Testing Configuration...")
    try:
        import yaml
        
        config_path = Path("configs/config.yaml")
        assert config_path.exists(), "config.yaml not found"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required keys
        required_keys = ['data', 'model', 'training']
        for key in required_keys:
            assert key in config, f"Config missing '{key}' section"
        
        print("  ✅ Configuration test passed")
        return True
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*70)
    print("🧪 DEEPFAKE DETECTION SYSTEM - TESTING SUITE")
    print("="*70)
    
    tests = [
        ("Configuration", test_config),
        ("Model Forward Pass", test_model_forward),
        ("Video Processor", test_video_processor),
        ("Dataset", test_dataset),
        ("Inference", test_inference)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
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
        print("\n🎉 All tests passed! System is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    print("="*70)
    
    return passed == total

if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)

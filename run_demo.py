"""
Quick demo script to test deepfake detection on a single video

Usage:
    python run_demo.py <video_path>

Example:
    python run_demo.py data/test/sample.mp4
"""

import torch
from pathlib import Path
from src.data_processing.video_processor import VideoProcessor
from src.inference.detector import DeepfakeInference
import sys

def run_demo(video_path):
    """Quick demo to test a single video"""
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found at {video_path}")
        return
    
    # Check if model exists
    model_path = "models/checkpoints/best_model.pth"
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("\nPlease train a model first:")
        print("  python train_model.py")
        return
    
    print("="*70)
    print("🎭 DEEPFAKE DETECTION DEMO")
    print("="*70)
    print(f"\n📹 Analyzing video: {video_path}")
    print("⏳ Processing... (this may take a few seconds)")
    print()
    
    try:
        # Initialize processor and detector
        processor = VideoProcessor()
        detector = DeepfakeInference(model_path, processor)
        
        # Run detection
        result = detector.predict_video(video_path)
        
        # Display results
        print("="*70)
        print("🔍 DETECTION RESULTS")
        print("="*70)
        
        # Prediction with color
        prediction = result['prediction']
        if prediction == 'FAKE':
            print(f"🚨 Prediction: {prediction}")
        elif prediction == 'REAL':
            print(f"✅ Prediction: {prediction}")
        else:
            print(f"⚠️  Prediction: {prediction}")
        
        print(f"🎯 Confidence: {result['confidence']:.2%}")
        print(f"📊 Fake Probability: {result['fake_probability']:.2%}")
        print(f"📊 Real Probability: {result['real_probability']:.2%}")
        
        print("\n" + "-"*70)
        print(f"💡 {result['explanation']}")
        print("="*70)
        
        # Additional info
        print("\n📝 Additional Information:")
        print(f"  - Video: {Path(video_path).name}")
        print(f"  - Size: {Path(video_path).stat().st_size / (1024*1024):.2f} MB")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if result['confidence'] < 0.7:
            print("  ⚠️  Low confidence - manual review recommended")
        if result['prediction'] == 'FAKE' and result['confidence'] > 0.8:
            print("  🚨 High confidence fake - strong manipulation detected")
        elif result['prediction'] == 'REAL' and result['confidence'] > 0.8:
            print("  ✅ High confidence real - appears authentic")
        
        print()
        
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("="*70)
        print("🎭 DEEPFAKE DETECTION DEMO")
        print("="*70)
        print("\nUsage:")
        print("  python run_demo.py <video_path>")
        print("\nExamples:")
        print("  python run_demo.py data/test/sample.mp4")
        print("  python run_demo.py path/to/your/video.mp4")
        print("\nSupported formats: MP4, AVI, MOV, MKV")
        print("="*70)
        
        # Optionally prompt for input
        video_path = input("\nEnter video path (or press Enter to exit): ").strip()
        if video_path:
            run_demo(video_path)
    else:
        run_demo(sys.argv[1])

if __name__ == "__main__":
    main()

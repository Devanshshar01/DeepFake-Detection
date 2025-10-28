# Training Guide: Achieving 80-90% Accuracy

This guide will help you train the deepfake detection model to achieve 80-90% accuracy.

## What Has Been Improved

### 1. Enhanced Model Architecture
- Upgraded from EfficientNet-B3 to EfficientNet-B4 backbone
- Added Channel and Spatial Attention mechanisms for better feature focus
- Implemented Multi-head Self-Attention for fusion layer
- Increased model capacity with deeper audio CNN (512 channels)
- Enhanced temporal modeling with 3-layer bidirectional LSTM
- Added residual connections in fusion layer

### 2. Advanced Training Strategies
- **Mixup Augmentation**: Blends training samples to improve generalization
- **Label Smoothing**: Reduces overconfidence and improves calibration
- **Cosine Annealing with Warm Restarts**: Better learning rate scheduling
- **Warmup Epochs**: Gradual learning rate increase at start
- **Focal Loss**: Optional loss function for handling class imbalance
- **Gradient Clipping**: Prevents exploding gradients

### 3. Comprehensive Data Augmentation
- Color transformations (brightness, contrast, hue, saturation)
- Blur variations (Gaussian, motion, median)
- Noise injection (Gaussian, ISO, multiplicative)
- Geometric transformations (rotation, scaling, shifts)
- Elastic deformations and distortions
- Cutout/CoarseDropout for regularization
- JPEG compression simulation
- Random cropping and resizing

### 4. Optimized Hyperparameters
- Increased epochs to 100 with early stopping at 15 patience
- Reduced batch size to 4 for better gradient estimates
- Increased frames per video to 20 for better temporal coverage
- Higher initial learning rate (0.0003) with cosine decay
- Lower weight decay (0.00001) to prevent underfitting

## Dataset Requirements

To achieve 80-90% accuracy, you need a substantial and diverse dataset:

### Recommended Dataset Size
- **Minimum**: 1,000 videos (500 real + 500 fake)
- **Good**: 5,000 videos (2,500 real + 2,500 fake)
- **Excellent**: 10,000+ videos (balanced between real and fake)

### Dataset Structure
```
data/
├── train/
│   ├── real/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── fake/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
└── val/
    ├── real/
    │   ├── video1.mp4
    │   └── ...
    └── fake/
        ├── video1.mp4
        └── ...
```

### Recommended Datasets
1. **FaceForensics++** (Highly Recommended)
   - Contains 1,000 original videos and 4,000 manipulated videos
   - Multiple deepfake methods included
   - Download: https://github.com/ondyari/FaceForensics

2. **Celeb-DF v2**
   - 590 real and 5,639 fake celebrity videos
   - High quality deepfakes
   - Download: https://github.com/yuezunli/celeb-deepfakeforensics

3. **DFDC (Deepfake Detection Challenge)**
   - 100,000+ videos
   - Diverse scenarios and subjects
   - Download: https://ai.facebook.com/datasets/dfdc/

### Data Split Recommendation
- Training: 80%
- Validation: 20%
- Ensure balanced distribution of real/fake in both sets

## Training Steps

### 1. Prepare Your Dataset
```bash
# Create directory structure
mkdir -p data/train/real data/train/fake data/val/real data/val/fake

# Place your videos in the appropriate folders
# Ensure you have at least 1,000+ videos total
```

### 2. Verify Configuration
Check `configs/config.yaml` to ensure settings are optimal:
- `batch_size: 4` (adjust based on GPU memory)
- `num_frames: 20`
- `epochs: 100`
- `learning_rate: 0.0003`
- `mixup_alpha: 0.2`
- `label_smoothing: 0.1`

### 3. Start Training
```bash
python train_model.py
```

### 4. Monitor Training
The training will display:
- Training and validation loss
- Training and validation accuracy
- Learning rate
- Patience counter for early stopping

Example output:
```
============================================================
Epoch 1/100
============================================================
Training (Epoch 1): 100%|████████| loss: 0.5234, acc: 75.23%, lr: 0.000060
Validation: 100%|████████████████|

Results:
  Train Loss: 0.5234 | Train Acc: 75.23%
  Val Loss:   0.4567 | Val Acc:   78.45%
  Learning Rate: 0.000060
  ✓ New best model! Accuracy: 78.45%
```

### 5. Training Progress Tracking
- Best model is saved to `models/checkpoints/best_model.pth`
- Training history is saved to `models/checkpoints/training_history.json`
- Checkpoint every 5 epochs saved as `checkpoint_epoch_N.pth`

## Expected Training Time

Training time varies based on hardware and dataset size:

| Hardware | Dataset Size | Estimated Time |
|----------|-------------|----------------|
| CPU | 1,000 videos | 20-40 hours |
| GPU (GTX 1080) | 1,000 videos | 3-6 hours |
| GPU (RTX 3090) | 1,000 videos | 1-2 hours |
| GPU (RTX 3090) | 10,000 videos | 10-20 hours |

## Tips for Achieving 80-90% Accuracy

### 1. Dataset Quality
- Use high-quality videos (720p or higher)
- Include diverse subjects, lighting, and backgrounds
- Mix different deepfake generation methods
- Balance real and fake samples (50/50 ratio)

### 2. Training Best Practices
- Start with default hyperparameters
- Monitor validation accuracy closely
- If overfitting (train acc >> val acc):
  - Increase dropout
  - Increase weight decay
  - Add more data augmentation
- If underfitting (both accuracies low):
  - Decrease dropout
  - Decrease weight decay
  - Train longer
  - Increase learning rate slightly

### 3. Hardware Recommendations
- **GPU**: Highly recommended (CUDA-capable NVIDIA GPU)
- **RAM**: 16GB minimum, 32GB recommended
- **VRAM**: 8GB minimum for batch_size=4
- **Storage**: SSD recommended for faster data loading

### 4. Debugging Low Accuracy

If accuracy is below 80% after training:

**Check 1: Dataset Balance**
```bash
# Count videos in each category
find data/train/real -type f | wc -l
find data/train/fake -type f | wc -l
```

**Check 2: Video Quality**
- Ensure videos contain clear faces
- Videos should be at least 1 second long
- Audio should be present

**Check 3: Model Loading**
- Verify the model is using GPU if available
- Check for any error messages during training

**Check 4: Learning Rate**
- If loss isn't decreasing, learning rate may be too low/high
- Try adjusting in config.yaml

### 5. Advanced Tuning

If you want to push beyond 90% accuracy:

1. **Use Focal Loss** (add to config.yaml):
   ```yaml
   training:
     use_focal_loss: true
   ```

2. **Increase Model Capacity** (edit `src/models/deepfake_detector.py`):
   - Use EfficientNet-B5 or B6
   - Increase hidden dimensions

3. **Ensemble Methods**:
   - Train multiple models with different seeds
   - Average predictions at inference time

4. **Data Quality**:
   - Manually review and remove low-quality samples
   - Add more diverse data sources

## Viewing Training Results

### In Terminal
Training progress is displayed in real-time with progress bars and metrics.

### In Streamlit UI
After training:
```bash
streamlit run app/streamlit_app.py
```

Navigate to the "Training Metrics" tab to see:
- Loss curves over epochs
- Accuracy curves over epochs
- Best validation accuracy achieved

## Troubleshooting

### Out of Memory Errors
```yaml
# Reduce batch size in configs/config.yaml
batch_size: 2  # or even 1
num_frames: 16  # reduce from 20
```

### Training Too Slow
```yaml
# Increase batch size if you have GPU memory
batch_size: 8
num_workers: 8  # for faster data loading
```

### Model Not Improving
- Check if dataset is too small (need 1,000+ videos minimum)
- Verify videos contain faces
- Ensure balanced dataset (equal real/fake)
- Try training longer (increase epochs)

### NaN Loss
- Reduce learning rate: `learning_rate: 0.0001`
- Ensure data is normalized properly
- Check for corrupted video files

## Next Steps After Training

1. **Evaluate Performance**:
   - Check validation accuracy in training logs
   - Review training history in Streamlit UI

2. **Test on New Videos**:
   - Use the Streamlit interface to test individual videos
   - Verify predictions make sense

3. **Fine-tune if Needed**:
   - If accuracy is 75-80%, train for more epochs
   - If accuracy is below 75%, review dataset quality
   - If accuracy is above 90%, congratulations!

4. **Deploy**:
   - Once satisfied with accuracy, deploy your application
   - Share with users or integrate into your workflow

## Summary

With the enhanced architecture, advanced training strategies, and comprehensive augmentations, your model is now capable of achieving 80-90% accuracy given:

1. A quality dataset of 1,000+ balanced videos
2. Proper GPU hardware
3. Following the training guide above
4. Monitoring and adjusting based on training progress

Good luck with your training!

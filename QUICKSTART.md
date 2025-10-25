# 🚀 Quick Start Guide

## Installation (2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install FFmpeg (macOS)
brew install ffmpeg

# 3. Create necessary directories
mkdir -p data/{train,val}/{real,fake}
mkdir -p models/checkpoints
```

## Try the Demo App Immediately

```bash
streamlit run app/streamlit_app.py
```

**Note**: The app will work but you need a trained model for predictions. See training section below.

## Train Your Model (Optional)

### Step 1: Get a Dataset

Download one of these datasets:
- **FaceForensics++**: https://github.com/ondyari/FaceForensics
- **Celeb-DF**: https://github.com/yuezunli/celeb-deepfakeforensics

### Step 2: Organize Your Data

```
data/
├── train/
│   ├── real/      # Put real videos here
│   └── fake/      # Put fake videos here
└── val/
    ├── real/
    └── fake/
```

### Step 3: Train

```python
import yaml
from torch.utils.data import DataLoader
from src.models.deepfake_detector import DeepfakeDetector
from src.data_processing.video_processor import VideoProcessor
from src.data_processing.dataset import DeepfakeDataset
from src.training.train import Trainer

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Prepare data (you'll need to customize this part)
processor = VideoProcessor()

# Example: Load your video paths and labels
train_videos = []  # List of video paths
train_labels = []  # List of labels (0=real, 1=fake)
val_videos = []
val_labels = []

train_dataset = DeepfakeDataset(train_videos, train_labels, processor)
val_dataset = DeepfakeDataset(val_videos, val_labels, processor)

train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'])

# Train
model = DeepfakeDetector()
trainer = Trainer(model, train_loader, val_loader, config)
trainer.train(config['training']['epochs'])
```

## Test Your Model

```python
from src.data_processing.video_processor import VideoProcessor
from src.inference.detector import DeepfakeInference

processor = VideoProcessor()
detector = DeepfakeInference("models/checkpoints/best_model.pth", processor)

results = detector.predict_video("path/to/test/video.mp4")
print(f"Prediction: {results['prediction']}")
print(f"Confidence: {results['confidence']:.2%}")
```

## 🎯 Key Files to Customize

1. **configs/config.yaml** - Adjust hyperparameters
2. **src/training/train.py** - Modify training loop if needed
3. **src/models/deepfake_detector.py** - Change model architecture
4. **app/streamlit_app.py** - Customize the web interface

## 📊 Expected Results

After training on a good dataset (10k+ videos):
- Training time: ~24 hours on single GPU
- Accuracy: 85-90%
- Model size: ~150MB

## 🐛 Troubleshooting

### "FFmpeg not found"
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt-get install ffmpeg
```

### "CUDA out of memory"
Reduce `batch_size` in `configs/config.yaml` from 8 to 4 or 2.

### "No module named 'src'"
Make sure you're running commands from the project root directory.

## 💡 Tips

1. **Start small**: Test with 100 videos before training on full dataset
2. **Use GPU**: Training on CPU will take days instead of hours
3. **Augmentation**: Enable data augmentation for better generalization
4. **Checkpoints**: Model saves automatically to `models/checkpoints/`

## 📚 Next Steps

- Read the full [README.md](README.md)
- Check out the [architecture documentation](README.md#architecture)
- Experiment with different model backbones
- Try ensemble methods for better accuracy

Happy detecting! 🎭

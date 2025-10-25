#!/bin/bash

# Deepfake Detector Quick Start Script
# This script sets up the environment for the deepfake detection system

echo "=================================="
echo "🎭 Deepfake Detector Quick Start"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
echo "   (This may take a few minutes...)"
pip install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check FFmpeg
echo ""
echo "🔍 Checking for FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "✓ FFmpeg found: $(ffmpeg -version | head -n 1)"
else
    echo "⚠️  FFmpeg not found"
    echo "   Install with: brew install ffmpeg (macOS)"
    echo "   or: sudo apt-get install ffmpeg (Ubuntu/Debian)"
fi

# Create necessary directories
echo ""
echo "📁 Creating necessary directories..."
mkdir -p data/raw/real data/raw/fake
mkdir -p data/train/real data/train/fake
mkdir -p data/val/real data/val/fake
mkdir -p data/test/real data/test/fake
mkdir -p models/checkpoints models/pretrained
mkdir -p evaluation_results logs

echo "✓ Directory structure created"

# Run installation test
echo ""
echo "🧪 Testing installation..."
python3 test_installation.py

# Print next steps
echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Get training data:"
echo "   → Run: python3 download_sample_data.py"
echo "   → Download FaceForensics++ or Celeb-DF"
echo ""
echo "2. Organize your dataset:"
echo "   → Place videos in data/raw/real/ and data/raw/fake/"
echo "   → Run: python3 prepare_dataset.py"
echo ""
echo "3. Train the model:"
echo "   → Run: python3 train_model.py"
echo "   → Training will use Apple Silicon GPU (MPS) automatically"
echo ""
echo "4. Test the system:"
echo "   → Run: python3 run_demo.py <video_path>"
echo ""
echo "5. Launch web application:"
echo "   → Run: python3 -m streamlit run app/streamlit_app.py"
echo "   → Open: http://localhost:8501"
echo ""
echo "=================================="
echo ""
echo "💡 Tip: Keep the virtual environment activated"
echo "   Activate: source venv/bin/activate"
echo "   Deactivate: deactivate"
echo ""
echo "📚 Documentation:"
echo "   - README.md for full documentation"
echo "   - USAGE_GUIDE.md for detailed instructions"
echo "   - STATUS.md for current system status"
echo ""
echo "🎉 Happy deepfake detecting!"
echo "=================================="

# 🚀 Deployment Guide for Deepfake Detection System

## ❌ Why Not Netlify?

Netlify is for **static sites** (HTML/CSS/JavaScript). Your project is a **Python ML application** that requires:
- Python runtime environment
- PyTorch and ML libraries (1.8GB+)
- Server-side processing
- GPU/CPU for model inference

## ✅ Recommended Deployment Platforms

---

## Option 1: Streamlit Cloud ⭐ (RECOMMENDED - Free & Easy)

**Perfect for your Streamlit app!**

### Setup Steps:

1. **Add deployment files** (already created):
   - `.streamlit/config.toml` ✅
   - `packages.txt` ✅
   - `requirements.txt` ✅

2. **Deploy**:
   - Go to: https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"
   - Select repo: `Devanshshar01/DeepFake-Detection`
   - Main file: `app/streamlit_app.py`
   - Click "Deploy"

3. **Your app will be live at**:
   ```
   https://your-app-name.streamlit.app
   ```

### Pros:
- ✅ Free tier available
- ✅ Perfect for Streamlit apps
- ✅ Automatic HTTPS
- ✅ Easy to use
- ✅ GitHub integration

### Cons:
- ⚠️ Limited CPU resources
- ⚠️ No GPU on free tier
- ⚠️ Memory limits (1GB)

---

## Option 2: Hugging Face Spaces ⭐ (Free GPU!)

**Best for ML demos with GPU support**

### Setup Steps:

1. **Create a Space**:
   - Go to: https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Streamlit" as SDK
   - Clone your repo into the Space

2. **Add configuration**:
   Create `README.md` in Space with:
   ```yaml
   ---
   title: Deepfake Detection
   emoji: 🎭
   colorFrom: blue
   colorTo: red
   sdk: streamlit
   sdk_version: 1.28.0
   app_file: app/streamlit_app.py
   pinned: false
   ---
   ```

3. **Push to Space**:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detection
   git push hf main
   ```

### Pros:
- ✅ Free GPU available
- ✅ Perfect for ML models
- ✅ Good for demos
- ✅ Community support

### Cons:
- ⚠️ More complex setup
- ⚠️ GPU may sleep after inactivity

---

## Option 3: Google Cloud Run (Scalable)

**Best for production deployments**

### Setup Steps:

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8080"]
   ```

2. **Deploy**:
   ```bash
   gcloud run deploy deepfake-detector \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

### Pros:
- ✅ Auto-scaling
- ✅ Pay per use
- ✅ Production-ready
- ✅ Custom domains

### Cons:
- 💰 Costs money (free tier available)
- ⚠️ More complex

---

## Option 4: AWS EC2 (Full Control)

**Best for complete control**

### Setup Steps:

1. **Launch EC2 instance** (t2.medium or larger)
2. **SSH and setup**:
   ```bash
   sudo apt update
   sudo apt install python3-pip
   git clone https://github.com/Devanshshar01/DeepFake-Detection.git
   cd DeepFake-Detection
   pip3 install -r requirements.txt
   streamlit run app/streamlit_app.py
   ```

3. **Configure security groups** to allow port 8501

### Pros:
- ✅ Full control
- ✅ Can use GPU instances
- ✅ Scalable

### Cons:
- 💰 Costs money
- ⚠️ Requires DevOps knowledge
- ⚠️ Manual setup

---

## Option 5: Heroku (Easy Deploy)

**Good for simple deployments**

### Setup Steps:

1. **Create files**:

`Procfile`:
```
web: streamlit run app/streamlit_app.py --server.port=$PORT
```

`runtime.txt`:
```
python-3.11.0
```

2. **Deploy**:
```bash
heroku create deepfake-detector
git push heroku main
```

### Pros:
- ✅ Easy to deploy
- ✅ GitHub integration
- ✅ Free tier

### Cons:
- 💰 Free tier limited
- ⚠️ Can be slow
- ⚠️ Sleeps after inactivity

---

## 🎯 Quick Start: Deploy to Streamlit Cloud (5 minutes)

### Step 1: Push deployment files
```bash
cd /Users/sharma.shruti@zomato.com/DeepFake-Detection
git add .streamlit/config.toml packages.txt
git commit -m "Add Streamlit Cloud deployment config"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Visit: https://streamlit.io/cloud
2. Sign in with GitHub
3. New app → Select your repo
4. Main file: `app/streamlit_app.py`
5. Deploy!

### Step 3: Wait for build (5-10 minutes)
Your app will be live at: `https://your-app.streamlit.app`

---

## 📊 Platform Comparison

| Platform | Cost | Setup | GPU | Best For |
|----------|------|-------|-----|----------|
| **Streamlit Cloud** | Free | ⭐⭐⭐⭐⭐ | ❌ | Quick demos |
| **Hugging Face** | Free | ⭐⭐⭐⭐ | ✅ | ML models |
| **Google Cloud Run** | $ | ⭐⭐⭐ | ❌ | Production |
| **AWS EC2** | $$ | ⭐⭐ | ✅ | Full control |
| **Heroku** | Free/$ | ⭐⭐⭐⭐ | ❌ | Simple apps |

---

## 🛠️ Optimization for Cloud Deployment

### Reduce Dependencies (Optional)
If deployment is slow, create a lighter `requirements-cloud.txt`:

```txt
streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python-headless>=4.8.0  # Lighter than opencv-python
timm>=0.9.0
numpy>=1.24.0
pillow>=10.0.0
pyyaml>=6.0.0
```

Then use: `pip install -r requirements-cloud.txt`

---

## 🎉 Recommended: Start with Streamlit Cloud

**It's the easiest and perfect for your use case!**

```bash
# Push config files
git add .streamlit/ packages.txt
git commit -m "Add Streamlit Cloud config"
git push origin main

# Then deploy on: https://streamlit.io/cloud
```

Your app will be live in minutes! 🚀

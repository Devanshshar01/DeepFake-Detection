import streamlit as st
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.video_processor import VideoProcessor
from src.inference.detector import DeepfakeInference

# Page config
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🎭",
    layout="wide"
)

# Initialize
@st.cache_resource
def load_detector():
    processor = VideoProcessor()
    detector = DeepfakeInference(
        "models/checkpoints/best_model.pth",
        processor
    )
    return detector

# Title
st.title("🎭 AI Deepfake Detection System")
st.markdown("Upload a video to analyze if it's AI-generated or authentic")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system uses deep learning to detect AI-generated fake videos.
    
    **Analysis includes:**
    - Facial feature consistency
    - Temporal motion patterns
    - Audio-visual synchronization
    
    **Supported formats:** MP4, AVI, MOV, MKV
    """)
    
    st.info("⚠️ This tool is for educational purposes. Always verify results manually.")

# Main content
uploaded_file = st.file_uploader(
    "Choose a video file",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Upload a video file to analyze"
)

if uploaded_file is not None:
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 Uploaded Video")
        st.video(video_path)
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        with st.spinner("🔄 Analyzing video... This may take a minute."):
            try:
                detector = load_detector()
                results = detector.predict_video(video_path)
                
                # Display prediction
                if results['prediction'] == 'FAKE':
                    st.error(f"🚨 **{results['prediction']}** ({results['confidence']:.1%} confidence)")
                elif results['prediction'] == 'REAL':
                    st.success(f"✅ **{results['prediction']}** ({results['confidence']:.1%} confidence)")
                else:
                    st.warning(f"⚠️ **{results['prediction']}**")
                
                # Confidence metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Fake Probability", f"{results['fake_probability']:.1%}")
                with col_b:
                    st.metric("Real Probability", f"{results['real_probability']:.1%}")
                
                # Explanation
                st.info(results['explanation'])
                
                # Progress bar for confidence
                st.progress(results['confidence'])
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Make sure you have a trained model at models/checkpoints/best_model.pth")
    
    # Clean up
    Path(video_path).unlink(missing_ok=True)

else:
    st.info("👆 Upload a video file to begin analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with PyTorch, Streamlit, and ❤️ | For educational purposes only
</div>
""", unsafe_allow_html=True)

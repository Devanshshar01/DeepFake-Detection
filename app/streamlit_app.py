import streamlit as st
import tempfile
from pathlib import Path
import sys
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.video_processor import VideoProcessor
from src.inference.detector import DeepfakeInference

st.set_page_config(
    page_title="Deepfake Detector AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_custom_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            text-align: center;
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .fake-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 2rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            box-shadow: 0 6px 12px rgba(245, 87, 108, 0.3);
        }
        .real-card {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 2rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            box-shadow: 0 6px 12px rgba(79, 172, 254, 0.3);
        }
        .uncertain-card {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 2rem;
            border-radius: 12px;
            color: #333;
            text-align: center;
            box-shadow: 0 6px 12px rgba(252, 182, 159, 0.3);
        }
        .info-box {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        .stButton>button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    """Load the deepfake detector model"""
    try:
        processor = VideoProcessor()
        detector = DeepfakeInference(
            "models/checkpoints/best_model.pth",
            processor
        )
        return detector, None
    except Exception as e:
        return None, str(e)

def create_confidence_gauge(confidence, prediction):
    """Create a gauge chart for confidence visualization"""
    if prediction == 'FAKE':
        color_scale = [[0, "#00f2fe"], [1, "#f5576c"]]
    else:
        color_scale = [[0, "#f5576c"], [1, "#00f2fe"]]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#f0f0f0'},
                {'range': [50, 75], 'color': '#e0e0e0'},
                {'range': [75, 100], 'color': '#d0d0d0'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#333", 'family': "Arial"}
    )

    return fig

def create_probability_chart(fake_prob, real_prob):
    """Create a bar chart for probability comparison"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Real', 'Fake'],
            y=[real_prob * 100, fake_prob * 100],
            marker=dict(
                color=['#00f2fe', '#f5576c'],
                line=dict(color='rgb(8,48,107)', width=1.5)
            ),
            text=[f'{real_prob*100:.1f}%', f'{fake_prob*100:.1f}%'],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Classification Probabilities",
        xaxis_title="Category",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        height=350,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig

def display_training_history():
    """Display training history if available"""
    history_path = Path("models/checkpoints/training_history.json")
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)

        st.subheader("📊 Training Performance")

        col1, col2 = st.columns(2)

        with col1:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                y=history['train_loss'],
                mode='lines',
                name='Train Loss',
                line=dict(color='#667eea', width=2)
            ))
            fig_loss.add_trace(go.Scatter(
                y=history['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='#f5576c', width=2)
            ))
            fig_loss.update_layout(
                title="Loss Over Time",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=300
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        with col2:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                y=[acc * 100 for acc in history['train_acc']],
                mode='lines',
                name='Train Accuracy',
                line=dict(color='#667eea', width=2)
            ))
            fig_acc.add_trace(go.Scatter(
                y=[acc * 100 for acc in history['val_acc']],
                mode='lines',
                name='Validation Accuracy',
                line=dict(color='#f5576c', width=2)
            ))
            fig_acc.update_layout(
                title="Accuracy Over Time",
                xaxis_title="Epoch",
                yaxis_title="Accuracy (%)",
                height=300
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        best_acc = max(history['val_acc']) * 100
        st.success(f"🎯 Best Validation Accuracy: {best_acc:.2f}%")

load_custom_css()

st.markdown('<h1 class="main-header">🔍 AI Deepfake Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced deep learning-powered video authenticity analysis</p>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://images.pexels.com/photos/8728380/pexels-photo-8728380.jpeg?auto=compress&cs=tinysrgb&w=400", use_container_width=True)

    st.markdown("### 🎯 About This System")
    st.markdown("""
    This advanced AI system uses state-of-the-art deep learning to detect deepfake videos with high accuracy.

    **Technology Stack:**
    - EfficientNet-B4 backbone
    - Multi-modal analysis (video + audio)
    - Attention mechanisms
    - Temporal pattern detection

    **Analysis Features:**
    - Facial feature consistency
    - Temporal motion patterns
    - Audio-visual synchronization
    - Frame-level anomaly detection
    """)

    st.markdown("### 📁 Supported Formats")
    st.markdown("MP4, AVI, MOV, MKV")

    st.markdown("### ⚙️ Model Information")
    model_path = Path("models/checkpoints/best_model.pth")
    if model_path.exists():
        st.success("✓ Model loaded")
        model_size = model_path.stat().st_size / (1024 * 1024)
        st.info(f"Model size: {model_size:.1f} MB")
    else:
        st.error("✗ Model not found")
        st.warning("Please train the model first")

    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.caption("This tool is for educational and research purposes. Always verify results manually.")

tab1, tab2 = st.tabs(["🎬 Video Analysis", "📈 Training Metrics"])

with tab1:
    st.markdown("## Upload Video for Analysis")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze for deepfake detection"
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📹 Uploaded Video")
            st.video(video_path)

            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.caption(f"File size: {file_size:.2f} MB")

        with col2:
            st.markdown("### 🔍 Analysis Results")

            with st.spinner("🔄 Analyzing video... This may take a minute."):
                detector, error = load_detector()

                if error:
                    st.error(f"❌ Failed to load model: {error}")
                    st.info("Make sure you have trained the model and it's saved at `models/checkpoints/best_model.pth`")
                else:
                    try:
                        results = detector.predict_video(video_path)

                        if results['prediction'] == 'FAKE':
                            st.markdown(f"""
                            <div class="fake-card">
                                <h2>🚨 DEEPFAKE DETECTED</h2>
                                <h3>{results['confidence']:.1%} Confidence</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        elif results['prediction'] == 'REAL':
                            st.markdown(f"""
                            <div class="real-card">
                                <h2>✅ AUTHENTIC VIDEO</h2>
                                <h3>{results['confidence']:.1%} Confidence</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="uncertain-card">
                                <h2>⚠️ UNCERTAIN</h2>
                                <h3>Further analysis recommended</h3>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("---")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(
                                "Fake Probability",
                                f"{results['fake_probability']:.1%}",
                                delta=f"{results['fake_probability']*100-50:.1f}%",
                                delta_color="inverse"
                            )
                        with col_b:
                            st.metric(
                                "Real Probability",
                                f"{results['real_probability']:.1%}",
                                delta=f"{results['real_probability']*100-50:.1f}%",
                                delta_color="normal"
                            )

                        st.markdown("### 📊 Detailed Analysis")

                        col_c, col_d = st.columns(2)
                        with col_c:
                            gauge_fig = create_confidence_gauge(results['confidence'], results['prediction'])
                            st.plotly_chart(gauge_fig, use_container_width=True)

                        with col_d:
                            prob_fig = create_probability_chart(
                                results['fake_probability'],
                                results['real_probability']
                            )
                            st.plotly_chart(prob_fig, use_container_width=True)

                        st.markdown("### 💡 Explanation")
                        st.info(results['explanation'])

                        with st.expander("🔬 Technical Details"):
                            st.json({
                                "prediction": results['prediction'],
                                "confidence": f"{results['confidence']:.4f}",
                                "fake_probability": f"{results['fake_probability']:.4f}",
                                "real_probability": f"{results['real_probability']:.4f}",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })

                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.exception(e)

        Path(video_path).unlink(missing_ok=True)

    else:
        st.info("👆 Upload a video file to begin analysis")

        st.markdown("### 🎯 How It Works")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>1️⃣ Video Processing</h4>
                <p>Extract frames and analyze facial features</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>2️⃣ AI Analysis</h4>
                <p>Deep learning model processes visual and audio data</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="info-box">
                <h4>3️⃣ Results</h4>
                <p>Get confidence score and detailed explanation</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("## 📈 Model Training Metrics")
    display_training_history()

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem;'>
    <p style='color: #666; font-size: 0.9rem;'>
        Built with PyTorch, Streamlit, and cutting-edge AI technology<br>
        For educational and research purposes only
    </p>
</div>
""", unsafe_allow_html=True)

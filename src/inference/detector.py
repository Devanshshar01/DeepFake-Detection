import torch
import numpy as np
from typing import Dict
from pathlib import Path

class DeepfakeInference:
    """Inference pipeline for deepfake detection"""
    
    def __init__(self, model_path: str, processor, device: str = None):
        # Auto-detect best available device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        self.processor = processor
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model"""
        from src.models.deepfake_detector import DeepfakeDetector
        
        model = DeepfakeDetector()
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        return model
    
    def predict_video(self, video_path: str, num_frames: int = 16) -> Dict:
        """
        Predict if video is fake or real
        
        Returns dict with prediction, confidence, and analysis
        """
        try:
            # Extract features
            frames = self.processor.extract_frames(video_path, num_frames)
            audio_features = self.processor.extract_audio_features(video_path)
            
            # Prepare tensors
            from src.data_processing.dataset import DeepfakeDataset
            transform = DeepfakeDataset.get_default_transform()
            
            frames_list = [transform(image=frame)['image'] for frame in frames]
            frames_tensor = torch.stack(frames_list).unsqueeze(0).to(self.device)
            audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(frames_tensor, audio_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                fake_prob = probabilities[0, 1].item()
                real_prob = probabilities[0, 0].item()
            
            prediction = 'FAKE' if fake_prob > 0.5 else 'REAL'
            confidence = max(fake_prob, real_prob)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'fake_probability': fake_prob,
                'real_probability': real_prob,
                'explanation': self.generate_explanation(fake_prob)
            }
        
        except Exception as e:
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'fake_probability': 0.0,
                'real_probability': 0.0,
                'explanation': f'Error processing video: {str(e)}'
            }
    
    def generate_explanation(self, fake_prob: float) -> str:
        """Generate human-readable explanation"""
        if fake_prob > 0.8:
            return "HIGH CONFIDENCE FAKE: Strong AI manipulation detected."
        elif fake_prob > 0.6:
            return "LIKELY FAKE: Multiple suspicious patterns found."
        elif fake_prob > 0.4:
            return "UNCERTAIN: Manual review recommended."
        elif fake_prob > 0.2:
            return "LIKELY REAL: Few suspicious indicators."
        else:
            return "HIGH CONFIDENCE REAL: Video appears authentic."

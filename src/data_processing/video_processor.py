import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import librosa

# Try to import MediaPipe, but make it optional
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Face detection will be disabled.")

class VideoProcessor:
    """Process videos for deepfake detection"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=0.5
            )
        else:
            self.mp_face_detection = None
    
    def extract_frames(self, video_path: str, num_frames: int = 16) -> List[np.ndarray]:
        """Extract uniformly sampled frames from video"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            raise ValueError(f"Could not read video: {video_path}")
        
        # Sample frames uniformly
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Detect and crop face
                face_frame = self.detect_and_crop_face(frame)
                if face_frame is not None:
                    frames.append(face_frame)
        
        cap.release()
        
        # If not enough frames with faces, use original frames
        if len(frames) < num_frames:
            cap = cv2.VideoCapture(video_path)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, self.target_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            cap.release()
        
        return frames
    
    def detect_and_crop_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect face and crop to region"""
        if not self.mp_face_detection:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_detection.process(rgb_frame)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            h, w = frame.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)
            
            # Add margin
            margin = 0.2
            x = max(0, int(x - w_box * margin))
            y = max(0, int(y - h_box * margin))
            w_box = int(w_box * (1 + 2 * margin))
            h_box = int(h_box * (1 + 2 * margin))
            
            face = frame[y:y+h_box, x:x+w_box]
            if face.size > 0:
                face = cv2.resize(face, self.target_size)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                return face
        
        return None
    
    def extract_audio_features(self, video_path: str) -> np.ndarray:
        """Extract MFCC features from audio"""
        try:
            # Extract audio
            import subprocess
            audio_path = video_path.replace('.mp4', '_temp.wav')
            subprocess.run([
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', audio_path, '-y'
            ], capture_output=True)
            
            # Load and extract MFCC
            audio, sr = librosa.load(audio_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Cleanup
            Path(audio_path).unlink(missing_ok=True)
            
            # Pad or truncate to fixed length
            target_length = 100
            if mfcc.shape[1] < target_length:
                mfcc = np.pad(mfcc, ((0, 0), (0, target_length - mfcc.shape[1])))
            else:
                mfcc = mfcc[:, :target_length]
            
            return mfcc
        
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            # Return zeros if audio extraction fails
            return np.zeros((13, 100))

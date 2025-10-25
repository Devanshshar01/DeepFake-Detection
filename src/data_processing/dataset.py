import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepfakeDataset(Dataset):
    """PyTorch Dataset for deepfake detection"""
    
    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        processor,
        num_frames: int = 16,
        transform=None
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        self.num_frames = num_frames
        self.transform = transform or self.get_default_transform()
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Extract frames
            frames = self.processor.extract_frames(video_path, self.num_frames)
            
            # Extract audio
            audio_features = self.processor.extract_audio_features(video_path)
            
            # Apply transformations
            transformed_frames = []
            for frame in frames:
                if self.transform:
                    frame = self.transform(image=frame)['image']
                transformed_frames.append(frame)
            
            frames_tensor = torch.stack(transformed_frames)
            audio_tensor = torch.FloatTensor(audio_features)
            
            return frames_tensor, audio_tensor, label
        
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            # Return dummy data
            frames = torch.zeros((self.num_frames, 3, 224, 224))
            audio = torch.zeros((13, 100))
            return frames, audio, label
    
    @staticmethod
    def get_default_transform():
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

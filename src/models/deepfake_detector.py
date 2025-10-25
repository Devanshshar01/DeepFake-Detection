import torch
import torch.nn as nn
import timm

class DeepfakeDetector(nn.Module):
    """Multi-modal deepfake detection model"""
    
    def __init__(self, num_classes: int = 2):
        super(DeepfakeDetector, self).__init__()
        
        # Spatial stream: EfficientNet-B3 backbone
        self.spatial_backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=True,
            num_classes=512
        )
        
        # Temporal stream: Bidirectional LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Audio stream: 1D CNN
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(13, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (batch, num_frames, 3, H, W)
            audio: (batch, 13, time_steps)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size, num_frames = frames.shape[:2]
        
        # Extract spatial features from each frame
        frames_flat = frames.view(-1, 3, 224, 224)
        spatial_features = self.spatial_backbone(frames_flat)
        spatial_features = spatial_features.view(batch_size, num_frames, -1)
        
        # Temporal modeling with LSTM
        temporal_features, _ = self.temporal_lstm(spatial_features)
        temporal_features = temporal_features[:, -1, :]  # Take last hidden state
        
        # Audio features
        audio_features = self.audio_cnn(audio).squeeze(-1)
        
        # Combine all modalities
        combined = torch.cat([
            spatial_features.mean(dim=1),  # Average spatial features
            temporal_features,
            audio_features
        ], dim=1)
        
        # Final classification
        logits = self.classifier(combined)
        return logits

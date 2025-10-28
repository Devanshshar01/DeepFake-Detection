import torch
import torch.nn as nn
import timm
from torch.nn import functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out

class DeepfakeDetector(nn.Module):
    """Enhanced multi-modal deepfake detection model with attention mechanisms"""

    def __init__(self, num_classes: int = 2):
        super(DeepfakeDetector, self).__init__()

        # Spatial stream: EfficientNet-B4 backbone (upgraded from B3)
        self.spatial_backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=True,
            features_only=True
        )

        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.spatial_backbone(dummy)
            feature_dim = features[-1].shape[1]

        # Attention mechanisms
        self.channel_attention = ChannelAttention(feature_dim)
        self.spatial_attention = SpatialAttention(feature_dim)

        # Adaptive pooling and projection
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_projection = nn.Sequential(
            nn.Linear(feature_dim, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # Enhanced temporal stream: Multi-layer Bidirectional LSTM with attention
        self.temporal_lstm = nn.LSTM(
            input_size=768,
            hidden_size=384,
            num_layers=3,
            batch_first=True,
            dropout=0.4,
            bidirectional=True
        )

        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(768, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # Enhanced audio stream: Deeper 1D CNN with residual connections
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(13, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.audio_projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # Multi-head fusion with cross-attention
        fusion_dim = 768 + 768 + 256
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.3,
            batch_first=True
        )

        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
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

        # Extract spatial features from each frame with attention
        frames_flat = frames.view(-1, 3, 224, 224)
        feature_maps = self.spatial_backbone(frames_flat)[-1]

        # Apply attention mechanisms
        feature_maps = self.channel_attention(feature_maps)
        feature_maps = self.spatial_attention(feature_maps)

        # Pool and project spatial features
        spatial_features = self.spatial_pool(feature_maps).flatten(1)
        spatial_features = self.spatial_projection(spatial_features)
        spatial_features = spatial_features.view(batch_size, num_frames, -1)

        # Temporal modeling with LSTM
        temporal_features, _ = self.temporal_lstm(spatial_features)

        # Apply temporal attention
        attention_weights = F.softmax(
            self.temporal_attention(temporal_features).squeeze(-1), dim=1
        )
        temporal_features = torch.bmm(
            attention_weights.unsqueeze(1),
            temporal_features
        ).squeeze(1)

        # Enhanced audio features
        audio_features = self.audio_cnn(audio).squeeze(-1)
        audio_features = self.audio_projection(audio_features)

        # Combine all modalities
        spatial_pooled = spatial_features.mean(dim=1)
        combined = torch.cat([
            spatial_pooled,
            temporal_features,
            audio_features
        ], dim=1)

        # Apply fusion attention
        combined_unsqueezed = combined.unsqueeze(1)
        fused, _ = self.fusion_attention(
            combined_unsqueezed,
            combined_unsqueezed,
            combined_unsqueezed
        )
        fused = fused.squeeze(1)

        # Residual connection
        fused = fused + combined

        # Final classification
        logits = self.classifier(fused)
        return logits

import torch
import torch.nn as nn
from torchvision import models

class CompatibleCNNLSTM(nn.Module):
    """Model compatible with saved weights"""
    def __init__(self, num_classes=13, sequence_length=16, hidden_size=512):
        super(CompatibleCNNLSTM, self).__init__()
        
        # CNN backbone (ResNet50 without final layer)
        resnet = models.resnet50(pretrained=False)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature projection: 2048 -> 512
        self.feature_projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # LSTM: 512 -> 512 (bidirectional = 1024 output)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(1024, 8, dropout=0.1, batch_first=True)
        
        # Classifier to match saved weights: 1024 -> 512 -> 13
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),  # This matches saved weight shape [512, 1024]
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 13)     # This matches saved weight shape [13, 512]
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # CNN feature extraction
        x = x.view(batch_size * seq_len, c, h, w)
        cnn_features = self.cnn_backbone(x)
        cnn_features = cnn_features.view(batch_size * seq_len, -1)
        
        # Feature projection
        projected_features = self.feature_projection(cnn_features)
        projected_features = projected_features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(projected_features)
        
        # Attention
        attended_features, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled_features = torch.mean(attended_features, dim=1)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits
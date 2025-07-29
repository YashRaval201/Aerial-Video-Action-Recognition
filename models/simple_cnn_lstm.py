import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleCNNLSTM(nn.Module):
    def __init__(self, num_classes=13, sequence_length=16, hidden_size=512):
        super(SimpleCNNLSTM, self).__init__()
        
        # Use pretrained ResNet50 for better features
        resnet = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # Unfreeze more layers for better learning
        for param in list(self.cnn.parameters())[:-20]:
            param.requires_grad = False
        
        # Enhanced feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(2048, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Enhanced LSTM with attention
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # CNN features
        x = x.view(batch_size * seq_len, c, h, w)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size * seq_len, -1)
        
        # Project features
        projected = self.feature_proj(cnn_features)
        projected = projected.view(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(projected)
        
        # Self-attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine LSTM and attention outputs
        combined = lstm_out + attended
        
        # Global max and average pooling
        max_pooled = torch.max(combined, dim=1)[0]
        avg_pooled = torch.mean(combined, dim=1)
        pooled = max_pooled + avg_pooled
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
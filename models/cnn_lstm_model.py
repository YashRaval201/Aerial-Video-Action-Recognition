import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=False, freeze_layers=False):
        super(ResNet50Backbone, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Train all layers from scratch
        for param in self.features.parameters():
            param.requires_grad = True
        
        self.feature_dim = 2048
    
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        features = self.features(x)
        # features shape: (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)
        # features shape: (batch_size, 2048)
        return features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size * 2)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)
        attn_out = self.dropout(attn_out)
        
        # Global average pooling over sequence dimension
        output = torch.mean(attn_out, dim=1)
        # output shape: (batch_size, hidden_size * 2)
        
        return output

class CNNLSTMActionRecognizer(nn.Module):
    def __init__(self, num_classes=13, sequence_length=16, hidden_size=512, 
                 num_lstm_layers=2, dropout=0.4, use_attention=True):
        super(CNNLSTMActionRecognizer, self).__init__()
        
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.use_attention = use_attention
        
        # CNN Backbone - Train from scratch on your dataset
        self.cnn_backbone = ResNet50Backbone(pretrained=False, freeze_layers=False)
        cnn_feature_dim = self.cnn_backbone.feature_dim
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(cnn_feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.BatchNorm1d(hidden_size)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=sequence_length)
        
        # LSTM with attention
        if use_attention:
            self.temporal_model = AttentionLSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_lstm_layers,
                dropout=dropout
            )
            lstm_output_dim = hidden_size * 2
        else:
            self.temporal_model = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
                dropout=dropout if num_lstm_layers > 1 else 0,
                bidirectional=True
            )
            lstm_output_dim = hidden_size * 2
        
        # Classification head with multiple layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape for CNN processing
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract CNN features
        cnn_features = self.cnn_backbone(x)
        # cnn_features shape: (batch_size * seq_len, feature_dim)
        
        # Project features
        projected_features = self.feature_projection(cnn_features)
        # projected_features shape: (batch_size * seq_len, hidden_size)
        
        # Reshape back to sequence format
        projected_features = projected_features.view(batch_size, seq_len, -1)
        # projected_features shape: (batch_size, seq_len, hidden_size)
        
        # Add positional encoding
        projected_features = projected_features.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        projected_features = self.pos_encoding(projected_features)
        projected_features = projected_features.transpose(0, 1)  # (batch_size, seq_len, hidden_size)
        
        # Temporal modeling
        if self.use_attention:
            temporal_features = self.temporal_model(projected_features)
        else:
            lstm_out, _ = self.temporal_model(projected_features)
            # Global average pooling
            temporal_features = torch.mean(lstm_out, dim=1)
        
        # Classification
        logits = self.classifier(temporal_features)
        
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
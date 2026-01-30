import torch
import torch.nn as nn
import torch.nn.functional as F
# Import ConvNeXtFeatureExtractor vừa thêm
from src.models.components import (
    ResNetFeatureExtractor, 
    ConvNeXtFeatureExtractor, 
    AttentionFusion, 
    PositionalEncoding, 
    STNBlock
)
import math

class ResTranOCR(nn.Module):
    """
    Modern OCR architecture using optional STN, ConvNeXt/ResNet and Transformer.
    """
    def __init__(
        self,
        num_classes: int,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True,
        backbone_type: str = "convnext" # Thêm option chọn backbone
    ):
        super().__init__()
        self.use_stn = use_stn
        self.backbone_type = backbone_type
        
        # 1. Spatial Transformer Network
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 2. Backbone Selection
        if backbone_type == "convnext":
            self.backbone = ConvNeXtFeatureExtractor()
            self.backbone_out_channels = 768 # ConvNeXt Tiny output
        else:
            self.backbone = ResNetFeatureExtractor(pretrained=False)
            self.backbone_out_channels = 512 # ResNet34 output
        
        # Project backbone features to a common dimension (512) for Fusion/Transformer
        self.feature_proj = nn.Conv2d(self.backbone_out_channels, 512, kernel_size=1)
        self.cnn_channels = 512
        
        # 3. Attention Fusion
        self.fusion = AttentionFusion(channels=self.cnn_channels)
        
        # 4. Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model=self.cnn_channels, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 5. Prediction Head
        self.head = nn.Linear(self.cnn_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)
        
        if self.use_stn:
            theta = self.stn(x_flat)
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat
        
        # Backbone features
        features = self.backbone(x_aligned)  # [B*F, Out_Ch, 1, W']
        
        # Projection if needed
        if features.size(1) != self.cnn_channels:
            features = self.feature_proj(features) # [B*F, 512, 1, W']
            
        fused = self.fusion(features)       # [B, 512, 1, W']
        
        # Prepare for Transformer: [B, C, 1, W'] -> [B, W', C]
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input) 
        
        out = self.head(seq_out)
        return out.log_softmax(2)
import torch
import torch.nn as nn
import torch.nn.functional as F
# Import ConvNeXtFeatureExtractor vừa thêm
from src.models.components import (
    ResNetFeatureExtractor, 
    ConvNeXtFeatureExtractor,
    TimmFeatureExtractor,
    AttentionFusion, 
    PositionalEncoding, 
    STNBlock,
    SRDecoder
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
        backbone_type: str = "convnext",
        backbone_pretrained: bool = False,
        timm_model: str = "",
        timm_out_index: int = 0,
        aux_sr: bool = False,
        cnn_channels: int = 512
    ):
        super().__init__()
        self.use_stn = use_stn
        self.aux_sr = aux_sr 
        
        # 1. Spatial Transformer Network
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 2. Backbone Selection
        if backbone_type == "timm":
            self.backbone = TimmFeatureExtractor(
                timm_model=timm_model,
                pretrained=bool(backbone_pretrained),
                out_index=int(timm_out_index),
            )
            self.backbone_out_channels = int(self.backbone.out_channels)
        elif backbone_type == "convnext":
            self.backbone = ConvNeXtFeatureExtractor()
            # try infer out_channels if implemented, else fall back to 768 (ConvNeXt Tiny)
            self.backbone_out_channels = int(getattr(self.backbone, "out_channels", 768))
        else:
            # torchvision resnet/resnext variants: resnet34/resnet50/resnet101/resnext50/resnext101/wide_resnet50
            arch = backbone_type
            if arch == "resnet":
                arch = "resnet34"
            self.backbone = ResNetFeatureExtractor(arch=arch, pretrained=bool(backbone_pretrained))
            self.backbone_out_channels = int(getattr(self.backbone, "out_channels", 512))

        
        # Project backbone features to a common dimension for Fusion/Transformer
        self.cnn_channels = int(cnn_channels)
        self.feature_proj = nn.Conv2d(self.backbone_out_channels, self.cnn_channels, kernel_size=1)
        
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

        # Thêm SR Decoder nếu bật flag
        if self.aux_sr:
            self.sr_decoder = SRDecoder(in_channels=self.cnn_channels)

    def forward(self, x, return_sr=False):
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)
        
        grid = None
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
            
        # Nhánh SR (Chạy nếu được yêu cầu)
        sr_out = None
        if self.aux_sr and return_sr:
            sr_out = self.sr_decoder(features)

        fused = self.fusion(features)       # [B, 512, 1, W']
        
        # Prepare for Transformer: [B, C, 1, W'] -> [B, W', C]
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input) 
        
        out = self.head(seq_out)
        
        log_probs = out.log_softmax(2)
        
        # Trả về thêm sr_out và grid để tính loss
        if return_sr:
            return log_probs, sr_out, grid
            
        return log_probs
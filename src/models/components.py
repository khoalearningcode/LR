import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ResNet34_Weights, resnet34,
    ResNet50_Weights, resnet50,
    ResNet101_Weights, resnet101,
    ResNeXt50_32X4D_Weights, resnext50_32x4d,
    ResNeXt101_32X8D_Weights, resnext101_32x8d,
    Wide_ResNet50_2_Weights, wide_resnet50_2,
)
import math

class LayerNorm2d(nn.Module):
    """LayerNorm that supports [N, C, H, W] inputs."""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from official implementation."""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()  # Placeholder, can add DropPath if needed

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtFeatureExtractor(nn.Module):
    """
    ConvNeXt Backbone customized for OCR (Height collapsing, Width preserving).
    Input: [N, 3, 32, 128]
    Output: [N, dims[-1], 1, 32] (Sequence length ~32)
    """
    def __init__(self, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        
        # Stem: 2x2 stride 2 (Not 4 to preserve details for LR images)
        # Input: 32x128 -> 16x64
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=2, stride=2),
            LayerNorm2d(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        # Stage 2 downsample: 2x2 stride 2
        # 16x64 -> 8x32
        downsample_layer1 = nn.Sequential(
            LayerNorm2d(dims[0], eps=1e-6),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
        )
        self.downsample_layers.append(downsample_layer1)

        # Stage 3 downsample: 2x1 stride (Collapse Height, Keep Width)
        # 8x32 -> 4x32
        downsample_layer2 = nn.Sequential(
            LayerNorm2d(dims[1], eps=1e-6),
            nn.Conv2d(dims[1], dims[2], kernel_size=(2, 1), stride=(2, 1)),
        )
        self.downsample_layers.append(downsample_layer2)

        # Stage 4 downsample: 2x1 stride
        # 4x32 -> 2x32
        downsample_layer3 = nn.Sequential(
            LayerNorm2d(dims[2], eps=1e-6),
            nn.Conv2d(dims[2], dims[3], kernel_size=(2, 1), stride=(2, 1)),
        )
        self.downsample_layers.append(downsample_layer3)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.final_norm = LayerNorm2d(dims[-1], eps=1e-6)
        
        # Final pooling to ensure height becomes 1
        # Input: 2x32 -> Output: 1x32
        self.final_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        # x: [B, 3, 32, 128]
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        x = self.final_norm(x) # [B, 768, 2, 32]
        x = self.final_pool(x) # [B, 768, 1, 32]
        return x

class STNBlock(nn.Module):
    """
    Spatial Transformer Network (STN) for image alignment.
    Learns to crop and rectify images before feeding them to the backbone.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Localization network: Predicts transformation parameters
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 8)) # Output fixed size for FC
        )
        
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [Batch, C, H, W]
        Returns:
            theta: Affine transformation matrix [Batch, 2, 3]
        """
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta


class AttentionFusion(nn.Module):
    """
    Attention-based fusion module for combining multi-frame features.
    Computes a weighted sum of features from multiple frames based on their 'quality' scores.
    """
    def __init__(self, channels: int):
        super().__init__()
        # A small CNN to predict attention scores (quality map) from features
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature maps from all frames. Shape: [Batch * Frames, C, H, W]
        Returns:
            Fused feature map. Shape: [Batch, C, H, W]
        """
        total_frames, c, h, w = x.size()
        num_frames = 5  # Fixed based on dataset
        batch_size = total_frames // num_frames

        # Reshape to [Batch, Frames, C, H, W]
        x_view = x.view(batch_size, num_frames, c, h, w)
        
        # Calculate attention scores: [Batch, Frames, 1, H, W]
        scores = self.score_net(x).view(batch_size, num_frames, 1, h, w)
        weights = F.softmax(scores, dim=1)  # Normalize scores across frames

        # Weighted sum fusion
        fused_features = torch.sum(x_view * weights, dim=1)
        return fused_features


class CNNBackbone(nn.Module):
    """A simple CNN backbone for CRNN baseline."""
    def __init__(self, out_channels=512):
        super().__init__()
        # Defined as a list of layers for clarity: Conv -> ReLU -> Pool
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 5 (Map to sequence height 1)
            nn.Conv2d(512, out_channels, 2, 1, 0), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )

    def forward(self, x):
        return self.features(x)


class ResNetFeatureExtractor(nn.Module):
    """ResNet-family backbones customized for OCR (preserve width, collapse height).

    Supported arch values:
      - resnet34, resnet50, resnet101
      - resnext50_32x4d, resnext101_32x8d
      - wide_resnet50_2
    """
    def __init__(self, arch: str = "resnet34", pretrained: bool = False):
        super().__init__()

        arch = arch.lower()
        if arch == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            net = resnet34(weights=weights)
            out_ch = 512
        elif arch == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            net = resnet50(weights=weights)
            out_ch = 2048
        elif arch == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            net = resnet101(weights=weights)
            out_ch = 2048
        elif arch in ("resnext50", "resnext50_32x4d"):
            weights = ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
            net = resnext50_32x4d(weights=weights)
            out_ch = 2048
        elif arch in ("resnext101", "resnext101_32x8d"):
            weights = ResNeXt101_32X8D_Weights.DEFAULT if pretrained else None
            net = resnext101_32x8d(weights=weights)
            out_ch = 2048
        elif arch in ("wide_resnet50", "wide_resnet50_2"):
            weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            net = wide_resnet50_2(weights=weights)
            out_ch = 2048
        else:
            raise ValueError(f"Unsupported ResNetFeatureExtractor arch: {arch}")

        self.out_channels = out_ch

        # --- OCR Customization ---
        # Keep width longer: modify layer3/layer4 strides to (2,1)
        # Works for ResNet/ResNeXt/WideResNet (same layer naming)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # Modify the first block stride in layer3 and layer4 to preserve width
        # layer3: stride (2,2) -> (2,1)
        if hasattr(self.layer3[0], "conv1"):
            self.layer3[0].conv1.stride = (2, 1)
        if hasattr(self.layer3[0], "downsample") and self.layer3[0].downsample is not None:
            self.layer3[0].downsample[0].stride = (2, 1)

        # layer4: stride (2,2) -> (2,1)
        if hasattr(self.layer4[0], "conv1"):
            self.layer4[0].conv1.stride = (2, 1)
        if hasattr(self.layer4[0], "downsample") and self.layer4[0].downsample is not None:
            self.layer4[0].downsample[0].stride = (2, 1)

        # Ensure final height becomes 1 (keep width)
        self.final_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return features as [B, C, 1, W']"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final_pool(x)
        return x
class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens in the sequence.
    Standard Sinusoidal implementation from 'Attention Is All You Need'.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [Batch, Seq_Len, Dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    

class SRDecoder(nn.Module):
    """
    Super-Resolution Head: Reconstructs image from features.
    Input: [Batch, 512, 1, 32] -> Output: [Batch, 3, 32, 128]
    """
    def __init__(self, in_channels=512):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Expand Height x4, Keep Width (H: 1->4, W: 32->32)
            nn.ConvTranspose2d(in_channels, 256, kernel_size=(4, 3), stride=(4, 1), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Layer 2: Expand Both x2 (H: 4->8, W: 32->64)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Layer 3: Expand Both x2 (H: 8->16, W: 64->128)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Layer 4 (SỬA Ở ĐÂY): Expand Height x2, Keep Width (H: 16->32, W: 128->128)
            # Dùng kernel (4,3), stride (2,1), padding (1,1) để giữ nguyên chiều rộng
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Final Conv
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh() # Output range [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

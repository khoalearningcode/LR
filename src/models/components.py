import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34
import math

class DropPath(nn.Module):
    """Stochastic Depth / DropPath (per-sample).

    This is a light-weight regularization that helps when scaling ConvNeXt depth.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        # Work with tensors of any dimensionality
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


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
        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0.0 else nn.Identity()

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
    def __init__(self, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate: float = 0.0, layer_scale_init_value: float = 1e-6):
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

        # Stochastic depth decay rule
        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, float(drop_path_rate), total_blocks).tolist() if total_blocks > 0 else []
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=float(dp_rates[cur + j]) if dp_rates else 0.0,
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
            cur += depths[i]
            self.stages.append(nn.Sequential(*blocks))

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
    def __init__(self, channels: int, num_frames: int = 5, frame_dropout: float = 0.0):
        super().__init__()
        self.num_frames = int(num_frames)
        self.frame_dropout = float(frame_dropout)

        # A small CNN to predict attention scores (quality map) from features
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // 8), 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature maps from all frames. Shape: [Batch * Frames, C, H, W]
        Returns:
            Fused feature map. Shape: [Batch, C, H, W]
        """
        total_frames, c, h, w = x.size()
        nf = self.num_frames
        assert total_frames % nf == 0, f"Expected total_frames % num_frames == 0, got {total_frames} vs {nf}"
        batch_size = total_frames // nf

        # Reshape to [Batch, Frames, C, H, W]
        x_view = x.view(batch_size, nf, c, h, w)

        # Scores: [Batch, Frames, 1, H, W]
        scores = self.score_net(x).view(batch_size, nf, 1, h, w)

        # Frame dropout (train-time only): force fusion to not overfit to one frame
        if self.training and self.frame_dropout > 0.0:
            keep = (torch.rand(batch_size, nf, device=x.device) >= self.frame_dropout)  # [B, F]
            # Ensure at least one frame kept for each sample
            none = ~keep.any(dim=1)
            if none.any():
                ridx = torch.randint(0, nf, (int(none.sum().item()),), device=x.device)
                keep[none, ridx] = True
            keep = keep.view(batch_size, nf, 1, 1, 1)
            scores = scores.masked_fill(~keep, -1e4)

        weights = F.softmax(scores, dim=1)
        fused_features = torch.sum(x_view * weights, dim=1)
        return fused_features


class TemporalTransformerFusion(nn.Module):
    """Temporal transformer across frames (per-timestep), then attention pooling.

    Input:  features from all frames  [B*F, C, 1, W]
    Output: fused features             [B,   C, 1, W]
    """
    def __init__(
        self,
        d_model: int,
        num_frames: int = 5,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        frame_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.frame_dropout = float(frame_dropout)

        if d_model % nhead != 0:
            raise ValueError(f"TemporalTransformerFusion: d_model ({d_model}) must be divisible by nhead ({nhead}).")

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_frames, c, h, w = x.size()
        nf = self.num_frames
        assert total_frames % nf == 0, f"Expected total_frames % num_frames == 0, got {total_frames} vs {nf}"
        b = total_frames // nf

        # [B, F, C, H, W] -> squeeze H (should be 1) -> [B, F, C, W]
        x = x.view(b, nf, c, h, w).squeeze(3)

        # Arrange as per-width timestep batches:
        # [B, F, C, W] -> [B, W, F, C] -> [B*W, F, C]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b * w, nf, c)

        src_key_padding_mask = None
        if self.training and self.frame_dropout > 0.0:
            keep = (torch.rand(b * w, nf, device=x.device) >= self.frame_dropout)
            none = ~keep.any(dim=1)
            if none.any():
                ridx = torch.randint(0, nf, (int(none.sum().item()),), device=x.device)
                keep[none, ridx] = True
            src_key_padding_mask = ~keep  # True means "masked / ignored" for TransformerEncoder

        x_enc = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B*W, F, C]

        scores = self.score(x_enc).squeeze(-1)  # [B*W, F]
        if src_key_padding_mask is not None:
            scores = scores.masked_fill(src_key_padding_mask, -1e4)

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B*W, F, 1]
        fused = (x_enc * weights).sum(dim=1)  # [B*W, C]

        # Back to feature map: [B*W, C] -> [B, C, 1, W]
        fused = fused.view(b, w, c).permute(0, 2, 1).unsqueeze(2).contiguous()
        return fused
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
    """
    ResNet-based backbone customized for OCR.
    Uses ResNet34 with modified strides to preserve width (sequence length) while reducing height.
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        
        # Load ResNet34 from torchvision
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = resnet34(weights=weights)

        # --- OCR Customization ---
        # We need to keep the standard first layer (stride 2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Modify strides in layer3 and layer4 to (2, 1)
        # This reduces height but preserves width for sequence modeling
        self.layer3[0].conv1.stride = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)
        
        self.layer4[0].conv1.stride = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [Batch, 3, H, W]
        Returns:
            Features [Batch, 512, H // 16, W // 2] (approx)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Ensure height is 1 for sequence modeling (Height collapsing)
        # Output shape: [Batch, 512, 1, W']
        x = F.adaptive_avg_pool2d(x, (1, None))
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
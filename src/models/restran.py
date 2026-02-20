import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import (
    TimmFeatureExtractor,   # <-- add

    ResNetFeatureExtractor,
    ConvNeXtFeatureExtractor,
    AttentionFusion,
    TemporalTransformerFusion,
    PositionalEncoding,
    STNBlock,
    SRDecoder,
)


class ResTranOCR(nn.Module):
    """
    Multi-frame OCR model:
      (5 frames) -> [optional STN] -> backbone (ConvNeXt/ResNet) -> feature proj (d_model)
      -> fusion across frames (attn OR temporal transformer)
      -> Transformer encoder over width -> CTC head

    New scaling knobs:
      - backbone_type: convnext_tiny/mid/small/base
      - drop_path_rate: stochastic depth for ConvNeXt
      - cnn_channels: d_model width (512 or 768 recommended)
      - fusion_type: "attn" or "temporal"
      - temporal_*: params for temporal fusion transformer (if enabled)
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
        aux_sr: bool = False,
        # ---- new knobs ----
        cnn_channels: int = 512,
        fusion_type: str = "attn",
        frame_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        backbone_pretrained: bool = False,  # only for resnet
        temporal_heads: int = 8,
        temporal_layers: int = 2,
        temporal_ff_dim: int = 1024,
        temporal_dropout: float = 0.1,
        num_frames: int = 5,
    ):
        super().__init__()
        self.use_stn = bool(use_stn)
        self.aux_sr = bool(aux_sr)
        self.num_frames = int(num_frames)

        self.cnn_channels = int(cnn_channels)
        self.fusion_type = str(fusion_type)

        # 1) STN
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 2) Backbone
        backbone_type = (backbone_type or "convnext").lower()

        # NEW: timm backbones, format:
        #   timm:<model_name>            (default out_index=0)
        #   timm:<model_name>:<out_idx>  (explicit stage)
        if backbone_type.startswith("timm:"):
            spec = backbone_type[len("timm:"):]  # e.g. "swinv2_base_window12_192.ms_in22k:0"
            out_index = 0
            model_name = spec
            # parse optional ":<digit>" at the end
            parts = spec.rsplit(":", 1)
            if len(parts) == 2 and parts[1].isdigit():
                model_name = parts[0]
                out_index = int(parts[1])

            # For timm, we WANT pretrained by default
            self.backbone = TimmFeatureExtractor(model_name=model_name, pretrained=True, in_chans=3, out_index=out_index)
            self.backbone_out_channels = int(self.backbone.out_channels)

        elif backbone_type in ["convnext", "convnext_tiny"]:
            depths = [3, 3, 9, 3]
            dims = [96, 192, 384, 768]
            self.backbone = ConvNeXtFeatureExtractor(depths=depths, dims=dims, drop_path_rate=float(drop_path_rate))
            self.backbone_out_channels = dims[-1]
        elif backbone_type in ["convnext_mid", "convnext_medium", "convnext_m"]:
            depths = [3, 3, 18, 3]
            dims = [96, 192, 384, 768]
            self.backbone = ConvNeXtFeatureExtractor(depths=depths, dims=dims, drop_path_rate=float(drop_path_rate))
            self.backbone_out_channels = dims[-1]
        elif backbone_type in ["convnext_small", "convnext_s"]:
            depths = [3, 3, 27, 3]
            dims = [96, 192, 384, 768]
            self.backbone = ConvNeXtFeatureExtractor(depths=depths, dims=dims, drop_path_rate=float(drop_path_rate))
            self.backbone_out_channels = dims[-1]
        elif backbone_type in ["convnext_base", "convnext_b"]:
            # Very heavy. Only try if you have enough GPU.
            depths = [3, 3, 27, 3]
            dims = [128, 256, 512, 1024]
            self.backbone = ConvNeXtFeatureExtractor(depths=depths, dims=dims, drop_path_rate=float(drop_path_rate))
            self.backbone_out_channels = dims[-1]
        else:
            # torchvision resnet/resnext/wide_resnet backbones (OCR stride customized)
            self.backbone = ResNetFeatureExtractor(arch=backbone_type, pretrained=bool(backbone_pretrained))
            self.backbone_out_channels = getattr(self.backbone, "out_channels", 512)

        # 3) Project backbone features -> d_model (cnn_channels)
        if self.backbone_out_channels == self.cnn_channels:
            self.feature_proj = nn.Identity()
        else:
            self.feature_proj = nn.Conv2d(self.backbone_out_channels, self.cnn_channels, kernel_size=1)

        # 4) Fusion across frames
        if self.fusion_type == "temporal":
            self.fusion = TemporalTransformerFusion(
                d_model=self.cnn_channels,
                num_frames=self.num_frames,
                nhead=int(temporal_heads),
                num_layers=int(temporal_layers),
                dim_feedforward=int(temporal_ff_dim),
                dropout=float(temporal_dropout),
                frame_dropout=float(frame_dropout),
            )
        else:
            # default: attention fusion
            self.fusion = AttentionFusion(
                channels=self.cnn_channels,
                num_frames=self.num_frames,
                frame_dropout=float(frame_dropout),
            )

        # 5) Transformer encoder over width
        if self.cnn_channels % int(transformer_heads) != 0:
            raise ValueError(
                f"ResTranOCR: cnn_channels ({self.cnn_channels}) must be divisible by transformer_heads ({transformer_heads})."
            )

        self.pos_encoder = PositionalEncoding(d_model=self.cnn_channels, dropout=float(dropout))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels,
            nhead=int(transformer_heads),
            dim_feedforward=int(transformer_ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(transformer_layers))

        # 6) CTC head
        self.head = nn.Linear(self.cnn_channels, num_classes)

        # 7) Optional SR decoder (aux task)
        if self.aux_sr:
            self.sr_decoder = SRDecoder(in_channels=self.cnn_channels)

    def forward(self, x: torch.Tensor, return_sr: bool = False):
        """
        Args:
            x: [B, F, 3, H, W]
        Returns:
            log_probs: [B, T, C]
            (optional) sr_out, grid
        """
        b, f, c, h, w = x.size()
        assert f == self.num_frames, f"Expected {self.num_frames} frames but got {f}"

        x_flat = x.view(b * f, c, h, w)

        grid = None
        if self.use_stn:
            theta = self.stn(x_flat)
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat

        # Backbone features: [B*F, Cb, 1, W']
        feat = self.backbone(x_aligned)

        # Project -> [B*F, d_model, 1, W']
        feat = self.feature_proj(feat)

        # Optional SR branch (per-frame)
        sr_out = None
        if self.aux_sr and return_sr:
            sr_out = self.sr_decoder(feat)

        # Fuse frames -> [B, d_model, 1, W']
        fused = self.fusion(feat)

        # Prepare for transformer: [B, d_model, 1, W'] -> [B, W', d_model]
        seq = fused.squeeze(2).permute(0, 2, 1)

        seq = self.pos_encoder(seq)
        seq = self.transformer(seq)

        logits = self.head(seq)
        log_probs = logits.log_softmax(2)

        if return_sr:
            return log_probs, sr_out, grid
        return log_probs
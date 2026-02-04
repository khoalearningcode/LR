"""Configuration dataclass for the training pipeline.

This project is for ICPR 2026 Competition – Low-Resolution License Plate Recognition (LRLPR).

Notes:
- All config fields are UPPERCASE so they can be serialized easily into config.json.
- We keep defaults aligned with your current best run (ResTran + STN), while adding
  flags to scale model capacity (ConvNeXt variants, drop-path, d_model, temporal fusion).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import torch


@dataclass
class Config:
    # ----------------------------
    # Experiment
    # ----------------------------
    MODEL_TYPE: str = "restran"  # "crnn" or "restran"
    EXPERIMENT_NAME: str = "restran"
    AUGMENTATION_LEVEL: str = "full"  # "full" or "light"
    USE_STN: bool = True

    # ----------------------------
    # Data paths
    # ----------------------------
    DATA_ROOT: str = "data/train"
    TEST_DATA_ROOT: str = "data/public_test"
    VAL_SPLIT_FILE: str = "data/val_tracks.json"
    SUBMISSION_FILE: str = "submission.txt"

    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128

    # ----------------------------
    # Charset
    # ----------------------------
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # ----------------------------
    # Training hyperparameters
    # ----------------------------
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 5e-4
    EPOCHS: int = 30
    SEED: int = 42
    NUM_WORKERS: int = 10
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 5.0
    SPLIT_RATIO: float = 0.9
    USE_CUDNN_BENCHMARK: bool = False

    # Gradient accumulation (helps train bigger models with small GPU RAM)
    GRAD_ACCUM_STEPS: int = 1  # 1 = disabled

    # Mild LR-simulation on REAL LR frames (train only). 0 = disabled
    TRAIN_LR_SIM_P: float = 0.0

    # ----------------------------
    # Backbone / Fusion / Model scaling
    # ----------------------------
    # Backbone: "resnet", "convnext"(=tiny), "convnext_tiny", "convnext_mid", "convnext_small", "convnext_base"
    BACKBONE_TYPE: str = "convnext"
    BACKBONE_PRETRAINED: bool = False  # only meaningful for resnet backbone
    DROPPATH_RATE: float = 0.0         # stochastic depth for ConvNeXt (0.0 ~ 0.2 typical)

    # Transformer/embedding width (d_model). Default 512 matches current code; try 768 for ConvNeXt.
    CNN_CHANNELS: int = 512

    # Multi-frame fusion
    FUSION_TYPE: str = "attn"       # "attn" | "temporal"
    FRAME_DROPOUT: float = 0.0      # drops some frames during training inside fusion

    # Temporal fusion (only used if FUSION_TYPE="temporal")
    TEMPORAL_HEADS: int = 8
    TEMPORAL_LAYERS: int = 2
    TEMPORAL_FF_DIM: int = 1024
    TEMPORAL_DROPOUT: float = 0.1

    # ----------------------------
    # CRNN model hyperparameters
    # ----------------------------
    HIDDEN_SIZE: int = 256
    RNN_DROPOUT: float = 0.25

    # ----------------------------
    # ResTranOCR hyperparameters
    # ----------------------------
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 3
    TRANSFORMER_FF_DIM: int = 2048
    TRANSFORMER_DROPOUT: float = 0.1

    # Optional auxiliary SR head
    AUX_SR: bool = False

    # ----------------------------
    # Runtime / output
    # ----------------------------
    DEVICE: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    OUTPUT_DIR: str = "results"

    # ----------------------------
    # Checkpointing / resume
    # ----------------------------
    CKPT_DIRNAME: str = "checkpoints"
    SAVE_EVERY_EPOCHS: int = 1   # save full ckpt "last.pt" every N epochs
    SAVE_EVERY_STEPS: int = 0    # if >0, also save mid-epoch every N minibatches
    KEEP_LAST_K: int = 2         # keep last K periodic checkpoints (optional)

    # ----------------------------
    # Derived attributes
    # ----------------------------
    CHAR2IDX: Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR: Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES: int = field(default=0, init=False)

    def __post_init__(self):
        self.CHAR2IDX = {char: idx + 1 for idx, char in enumerate(self.CHARS)}
        self.IDX2CHAR = {idx + 1: char for idx, char in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS) + 1  # +1 for blank


def get_default_config() -> Config:
    return Config()
#!/usr/bin/env python3
"""Main entry point for OCR training pipeline."""
import argparse
import os
import sys
import json
import csv
import time
import shlex
import socket
import platform
import subprocess
import math
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.training.trainer import Trainer
from src.utils.common import seed_everything


# ----------------------------
# Helpers: serialization + meta
# ----------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_json(obj):
    """Best-effort convert to JSON-serializable."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return sorted([_safe_json(x) for x in obj])
    # torch device, numpy types, etc.
    try:
        return obj.item()
    except Exception:
        pass
    return str(obj)


def dump_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_safe_json(data), f, ensure_ascii=False, indent=2)


def dump_csv(path: str, rows: list):
    """rows: list[dict]"""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # union all keys to avoid missing columns
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def collect_git_info(project_root: str) -> dict:
    def _run(cmd):
        try:
            out = subprocess.check_output(
                cmd, cwd=project_root, stderr=subprocess.DEVNULL
            ).decode("utf-8", errors="ignore").strip()
            return out
        except Exception:
            return None

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "is_dirty": (_run(["git", "status", "--porcelain"]) not in (None, "")),
    }


def collect_system_info() -> dict:
    cuda_available = torch.cuda.is_available()
    cuda_dev = None
    if cuda_available:
        try:
            cuda_dev = {
                "name": torch.cuda.get_device_name(0),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                "num_devices": torch.cuda.device_count(),
            }
        except Exception:
            cuda_dev = {"error": "failed to query cuda device info"}

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device": cuda_dev,
    }


def serialize_config(config: Config) -> dict:
    """Grab UPPERCASE attrs from config into a dict."""
    cfg = {}
    for k in dir(config):
        if not k.isupper():
            continue
        try:
            v = getattr(config, k)
        except Exception:
            continue
        cfg[k] = _safe_json(v)
    return cfg


class Tee:
    """Duplicate stdout/stderr to multiple file-like objects."""
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass


def write_summary_md(path: str, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append(f"# Run summary\n")
    lines.append(f"- run_id: `{meta.get('run_id')}`")
    lines.append(f"- started: `{meta.get('started_at')}`")
    lines.append(f"- ended: `{meta.get('ended_at')}`")
    lines.append(f"- duration_sec: `{meta.get('duration_sec')}`")
    lines.append(f"- command: `{meta.get('command')}`")
    lines.append("")

    cfg = meta.get("config", {})
    lines.append("## Config")
    for key in [
        "EXPERIMENT_NAME", "MODEL_TYPE", "USE_STN", "DATA_ROOT",
        "EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "IMG_HEIGHT", "IMG_WIDTH",
        "AUGMENTATION_LEVEL", "VAL_SPLIT_FILE", "DEVICE"
    ]:
        if key in cfg:
            lines.append(f"- {key}: `{cfg[key]}`")
    lines.append("")

    ds = meta.get("dataset", {})
    if ds:
        lines.append("## Dataset")
        for k, v in ds.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")

    model = meta.get("model", {})
    if model:
        lines.append("## Model")
        for k, v in model.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")

    best = meta.get("best", {})
    if best:
        lines.append("## Best checkpoint / metrics")
        for k, v in best.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")

    files = meta.get("artifacts", {})
    if files:
        lines.append("## Artifacts")
        for k, v in files.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Multi-Frame OCR for License Plate Recognition"
    )
    parser.add_argument(
        "-n", "--experiment-name", type=str, default=None,
        help="Experiment name for checkpoint/submission files (default: from config)"
    )
    parser.add_argument(
        "-m", "--model", type=str, choices=["crnn", "restran"], default=None,
        help="Model architecture: 'crnn' or 'restran' (default: from config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Total number of training epochs (default: from config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for training (default: from config)"
    )
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=None,
        dest="learning_rate",
        help="Learning rate (default: from config)"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root directory for training data (default: from config)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from config)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of data loader workers (default: from config)"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=None,
        help="LSTM hidden size for CRNN (default: from config)"
    )
    parser.add_argument(
        "--transformer-heads", type=int, default=None,
        help="Number of transformer attention heads (default: from config)"
    )
    parser.add_argument(
        "--transformer-layers", type=int, default=None,
        help="Number of transformer encoder layers (default: from config)"
    )

    parser.add_argument(
        "--transformer-ff-dim", type=int, default=None,
        help="Transformer FFN dimension (ResTran) (default: from config)"
    )
    parser.add_argument(
        "--transformer-dropout", type=float, default=None,
        help="Transformer dropout (ResTran) (default: from config)"
    )
    parser.add_argument(
        "--cnn-channels", type=int, default=None,
        help="Embedding width / d_model after backbone (512 or 768 recommended) (default: from config)"
    )
    parser.add_argument(
        "--drop-path-rate", type=float, default=None,
        help="ConvNeXt stochastic depth rate (0.0~0.2) (default: from config)"
    )
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["attn", "temporal"],
        default=None,
        help="Multi-frame fusion type: attn | temporal (default: from config)",
    )
    parser.add_argument("--temporal-heads", type=int, default=None, help="Temporal fusion transformer heads (default: from config)")
    parser.add_argument("--temporal-layers", type=int, default=None, help="Temporal fusion transformer layers (default: from config)")
    parser.add_argument("--temporal-ff-dim", type=int, default=None, help="Temporal fusion transformer FF dim (default: from config)")
    parser.add_argument("--temporal-dropout", type=float, default=None, help="Temporal fusion transformer dropout (default: from config)")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (>=1). Effective batch = batch_size * grad_accum_steps.",
    )
    parser.add_argument(
        "--aug-level",
        type=str,
        choices=["full", "light"],
        default=None,
        help="Augmentation level for training data (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Base directory to store runs (default: results/)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Stable run folder name/path (NO timestamp). If relative, it is under --output-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing run directory (only when NOT resuming).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last full checkpoint in the run directory.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Explicit path to a full checkpoint (.pt) to resume from.",
    )
    parser.add_argument(
        "--reset-optim",
        action="store_true",
        help="When resuming, reset optimizer state (load weights only).",
    )
    parser.add_argument(
        "--reset-scheduler",
        action="store_true",
        help="When resuming, reset LR scheduler state (required if batch-size/steps/epochs changed).",
    )
    parser.add_argument(
        "--reset-scaler",
        action="store_true",
        help="When resuming, reset AMP GradScaler state.",
    )
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=None,
        help="Save last checkpoint every N epochs (default: from config).",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=None,
        help="If >0, save last checkpoint every N train steps (slower).",
    )
    parser.add_argument(
        "--no-stn",
        action="store_true",
        help="Disable Spatial Transformer Network (STN) alignment",
    )
    parser.add_argument(
        "--submission-mode",
        action="store_true",
        help="Train on full dataset and generate submission file for test data",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional tag appended into run folder name when --run-dir is not set.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet", "convnext", "convnext_tiny", "convnext_mid", "convnext_small", "convnext_base"],
        default=None,
        help="Backbone architecture for ResTran: 'resnet' or 'convnext'",
    )
    parser.add_argument(
        "--sr-aux",
        action="store_true",
        help="Enable Super-Resolution Auxiliary Branch",
    )

    # ---- Stronger training/inference flags (optional) ----
    parser.add_argument("--img-height", type=int, default=None, help="Override image height (default: from config)")
    parser.add_argument("--img-width", type=int, default=None, help="Override image width (default: from config)")
    parser.add_argument(
        "--train-lr-sim-p",
        type=float,
        default=None,
        help="Probability of mild LR-like degradation on training images (requires patched transforms/dataset).",
    )
    parser.add_argument(
        "--frame-dropout",
        type=float,
        default=None,
        help="Frame dropout inside AttentionFusion during training (requires patched model).",
    )
    parser.add_argument(
        "--backbone-pretrained",
        action="store_true",
        help="Use pretrained weights for ResNet backbone (only if backbone=resnet and model supports it).",
    )
    return parser.parse_args()



def _resolve_run_dir(args: argparse.Namespace, exp_name: str) -> tuple[str, str]:
    """
    Returns (run_dir, run_name).
    - If --run-dir is provided:
        * if it's absolute or contains a path separator, use it as-is
        * else use <output-dir>/<run-dir>
    - Else: <output-dir>/<exp_name>[_run-tag]
    """
    if args.run_dir:
        run_dir = args.run_dir
        if not os.path.isabs(run_dir) and (os.sep not in run_dir):
            run_dir = os.path.join(args.output_dir, run_dir)
        run_dir = os.path.normpath(run_dir)
        run_name = os.path.basename(run_dir)
        return run_dir, run_name

    run_name = exp_name
    if args.run_tag:
        run_name += f"_{args.run_tag}"
    run_dir = os.path.join(args.output_dir, run_name)
    run_dir = os.path.normpath(run_dir)
    return run_dir, run_name


def _load_config_json_into_config(cfg_obj: Config, cfg_json_path: str) -> None:
    try:
        with open(cfg_json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return
    for k, v in d.items():
        if hasattr(cfg_obj, k):
            try:
                setattr(cfg_obj, k, v)
            except Exception:
                pass


def main():
    """Main training entry point (supports stable run dirs + resume)."""
    args = parse_args()

    # 1) Create base config and decide run directory (NO timestamp)
    config = Config()

    # Allow CLI exp_name to affect default run_dir naming
    if args.experiment_name is not None:
        config.EXPERIMENT_NAME = args.experiment_name

    run_dir, run_name = _resolve_run_dir(args, config.EXPERIMENT_NAME)

    # 2) Resume: load config.json from run_dir first (then CLI overrides win)
    cfg_path = os.path.join(run_dir, "config.json")
    run_meta_path = os.path.join(run_dir, "run_meta.json")

    if args.resume:
        if not os.path.exists(run_dir):
            print(f"❌ ERROR: --resume nhưng run_dir không tồn tại: {run_dir}")
            sys.exit(1)
        if os.path.exists(cfg_path):
            _load_config_json_into_config(config, cfg_path)

    # 3) Apply CLI overrides (wins over loaded config.json)
    arg_to_config = {
        "experiment_name": "EXPERIMENT_NAME",
        "model": "MODEL_TYPE",
        "epochs": "EPOCHS",
        "batch_size": "BATCH_SIZE",
        "learning_rate": "LEARNING_RATE",
        "data_root": "DATA_ROOT",
        "seed": "SEED",
        "num_workers": "NUM_WORKERS",
        "hidden_size": "HIDDEN_SIZE",
        "transformer_heads": "TRANSFORMER_HEADS",
        "transformer_layers": "TRANSFORMER_LAYERS",
        "transformer_ff_dim": "TRANSFORMER_FF_DIM",
        "transformer_dropout": "TRANSFORMER_DROPOUT",
        "cnn_channels": "CNN_CHANNELS",
        "drop_path_rate": "DROPPATH_RATE",
        "fusion": "FUSION_TYPE",
        "temporal_heads": "TEMPORAL_HEADS",
        "temporal_layers": "TEMPORAL_LAYERS",
        "temporal_ff_dim": "TEMPORAL_FF_DIM",
        "temporal_dropout": "TEMPORAL_DROPOUT",
        "grad_accum_steps": "GRAD_ACCUM_STEPS",
        "img_height": "IMG_HEIGHT",
        "img_width": "IMG_WIDTH",
    }
    for arg_name, config_name in arg_to_config.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)

    if args.aug_level is not None:
        config.AUGMENTATION_LEVEL = args.aug_level
    if args.no_stn:
        config.USE_STN = False
    if args.backbone is not None:
        config.BACKBONE_TYPE = args.backbone
    config.AUX_SR = bool(args.sr_aux)

    # ---- New flag overrides (safe even if Config didn't declare them) ----
    if args.img_height is not None:
        config.IMG_HEIGHT = int(args.img_height)
    if args.img_width is not None:
        config.IMG_WIDTH = int(args.img_width)

    if args.train_lr_sim_p is not None:
        setattr(config, "TRAIN_LR_SIM_P", float(args.train_lr_sim_p))
    if args.frame_dropout is not None:
        setattr(config, "FRAME_DROPOUT", float(args.frame_dropout))
    # store backbone_pretrained for model construction if supported
    setattr(config, "BACKBONE_PRETRAINED", bool(args.backbone_pretrained))

    # checkpoint frequency overrides
    if args.save_every_epochs is not None:
        config.SAVE_EVERY_EPOCHS = int(args.save_every_epochs)
    if args.save_every_steps is not None:
        config.SAVE_EVERY_STEPS = int(args.save_every_steps)

    # 4) Create/validate run directory
    if os.path.exists(run_dir) and (not args.resume):
        if args.overwrite:
            import shutil
            shutil.rmtree(run_dir, ignore_errors=True)
        else:
            print(f"❌ ERROR: run_dir đã tồn tại: {run_dir}")
            print("   - Dùng --overwrite để ghi đè, hoặc đặt --run-dir / --run-tag khác.")
            sys.exit(1)

    os.makedirs(run_dir, exist_ok=True)
    config.OUTPUT_DIR = run_dir

    # 5) Setup console logging
    log_path = os.path.join(run_dir, "console.log")
    _stdout = sys.stdout
    _stderr = sys.stderr
    log_f = open(log_path, "a" if args.resume else "w", encoding="utf-8")
    sys.stdout = Tee(_stdout, log_f)
    sys.stderr = Tee(_stderr, log_f)

    started_at = _now_iso()
    t0 = time.time()

    seed_everything(config.SEED)

    # 6) Run meta (append sessions if resuming)
    project_root = os.path.dirname(os.path.abspath(__file__))
    if args.resume and os.path.exists(run_meta_path):
        try:
            run_meta = json.load(open(run_meta_path, "r", encoding="utf-8"))
        except Exception:
            run_meta = {}
    else:
        run_meta = {}

    run_meta.setdefault("run_id", run_name)
    run_meta.setdefault("system", collect_system_info())
    run_meta.setdefault("git", collect_git_info(project_root))
    run_meta.setdefault("sessions", [])

    run_meta["args"] = vars(args)
    run_meta["config"] = serialize_config(config)
    run_meta["ended_at"] = None
    run_meta["duration_sec"] = None

    run_meta["sessions"].append({
        "started_at": started_at,
        "ended_at": None,
        "duration_sec": None,
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "resume": bool(args.resume),
        "resume_path": args.resume_path,
    })

    dump_json(run_meta_path, run_meta)
    dump_json(cfg_path, run_meta["config"])

    exp_name = config.EXPERIMENT_NAME

    print("🚀 Configuration:")
    print(f"   RUN_DIR        : {run_dir}")
    print(f"   EXPERIMENT     : {exp_name}")
    print(f"   MODEL          : {config.MODEL_TYPE}")
    print(f"   BACKBONE       : {getattr(config, 'BACKBONE_TYPE', None)}")
    print(f"   CNN_CHANNELS   : {getattr(config, 'CNN_CHANNELS', None)}")
    print(f"   FUSION         : {getattr(config, 'FUSION_TYPE', None)} | FD={getattr(config, 'FRAME_DROPOUT', None)}")
    print(f"   DROPPATH       : {getattr(config, 'DROPPATH_RATE', None)}")
    print(f"   GRAD_ACCUM     : {getattr(config, 'GRAD_ACCUM_STEPS', 1)}")
    print(f"   USE_STN        : {config.USE_STN}")
    print(f"   DATA_ROOT      : {config.DATA_ROOT}")
    print(f"   EPOCHS(total)  : {config.EPOCHS}")
    print(f"   BATCH_SIZE     : {config.BATCH_SIZE}")
    print(f"   LR             : {config.LEARNING_RATE}")
    print(f"   DEVICE         : {config.DEVICE}")
    print(f"   AUX_SR         : {config.AUX_SR}")
    print(f"   SAVE_EPOCHS    : {getattr(config, 'SAVE_EVERY_EPOCHS', 1)}")
    print(f"   SAVE_STEPS     : {getattr(config, 'SAVE_EVERY_STEPS', 0)}")
    print(f"   RESUME         : {bool(args.resume)}")
    print(f"   META           : {run_meta_path}")

    # Validate data path
    if not os.path.exists(config.DATA_ROOT):
        print(f"❌ ERROR: Data root not found: {config.DATA_ROOT}")
        sys.exit(1)

    # Common dataset parameters
    common_ds_params = {
        "split_ratio": config.SPLIT_RATIO,
        "img_height": config.IMG_HEIGHT,
        "img_width": config.IMG_WIDTH,
        "char2idx": config.CHAR2IDX,
        "val_split_file": config.VAL_SPLIT_FILE,
        "seed": config.SEED,
        "augmentation_level": config.AUGMENTATION_LEVEL,
    }

    # Optional: mild LR simulation probability for REAL LR frames (only if dataset supports it)
    try:
        import inspect
        if "train_lr_sim_p" in inspect.signature(MultiFrameDataset.__init__).parameters:
            common_ds_params["train_lr_sim_p"] = float(getattr(config, "TRAIN_LR_SIM_P", 0.0))
    except Exception:
        pass


    # Create datasets based on mode
    if args.submission_mode:
        print("\n📌 SUBMISSION MODE ENABLED")
        print("   - Training on FULL dataset (no validation split)")
        print("   - Will generate predictions for test data after training\n")

        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode="train",
            full_train=True,
            **common_ds_params,
        )

        test_loader = None
        if os.path.exists(config.TEST_DATA_ROOT):
            test_ds = MultiFrameDataset(
                root_dir=config.TEST_DATA_ROOT,
                mode="val",
                img_height=config.IMG_HEIGHT,
                img_width=config.IMG_WIDTH,
        input_norm=getattr(config,'INPUT_NORM','none'),
                char2idx=config.CHAR2IDX,
                seed=config.SEED,
                is_test=True,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True,
            )
        else:
            print(f"⚠️ WARNING: Test data not found at {config.TEST_DATA_ROOT}")

        val_loader = None
        val_ds = None
    else:
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode="train",
            **common_ds_params,
        )

        val_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode="val",
            **common_ds_params,
        )

        val_loader = None
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True,
            )
        else:
            print("⚠️ WARNING: Validation dataset is empty.")

        test_loader = None

    if len(train_ds) == 0:
        print("❌ Training dataset is empty!")
        sys.exit(1)

    run_meta["dataset"] = {
        "train_size": len(train_ds),
        "val_size": (len(val_ds) if val_ds is not None else 0),
        "submission_mode": bool(args.submission_mode),
        "data_root": config.DATA_ROOT,
        "test_data_root": getattr(config, "TEST_DATA_ROOT", None),
    }
    dump_json(run_meta_path, run_meta)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize model based on config

    # Helper: pass only kwargs supported by the target callable (keeps backward compatibility)
    def _filter_kwargs(callable_obj, kwargs: dict) -> dict:
        try:
            import inspect
            sig = inspect.signature(callable_obj)
            return {k: v for k, v in kwargs.items() if k in sig.parameters}
        except Exception:
            return kwargs

    if config.MODEL_TYPE == "restran":
        restran_kwargs = dict(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            backbone_type=config.BACKBONE_TYPE,
            aux_sr=config.AUX_SR,
            cnn_channels=int(getattr(config, "CNN_CHANNELS", 512)),
            fusion_type=str(getattr(config, "FUSION_TYPE", "attn")),
            drop_path_rate=float(getattr(config, "DROPPATH_RATE", 0.0)),
            temporal_heads=int(getattr(config, "TEMPORAL_HEADS", 8)),
            temporal_layers=int(getattr(config, "TEMPORAL_LAYERS", 2)),
            temporal_ff_dim=int(getattr(config, "TEMPORAL_FF_DIM", 1024)),
            temporal_dropout=float(getattr(config, "TEMPORAL_DROPOUT", 0.1)),

            # optional extras (only used if your ResTranOCR __init__ supports them)
            frame_dropout=float(getattr(config, "FRAME_DROPOUT", 0.0)),
            backbone_pretrained=bool(getattr(config, "BACKBONE_PRETRAINED", False)),
        )
        model = ResTranOCR(**_filter_kwargs(ResTranOCR, restran_kwargs)).to(config.DEVICE)
    else:
        crnn_kwargs = dict(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=config.USE_STN,
            frame_dropout=float(getattr(config, "FRAME_DROPOUT", 0.0)),
        )
        model = MultiFrameCRNN(**_filter_kwargs(MultiFrameCRNN, crnn_kwargs)).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model ({config.MODEL_TYPE}): {total_params:,} total params, {trainable_params:,} trainable")

    run_meta["model"] = {
        "type": config.MODEL_TYPE,
        "use_stn": bool(config.USE_STN),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }
    dump_json(run_meta_path, run_meta)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR,
    )

    # 7) Resume from checkpoint (FULL state)
    resumed_from = None
    if args.resume or args.resume_path:
        ckpt_path = args.resume_path
        if ckpt_path is None:
            ckpt_dirname = getattr(config, "CKPT_DIRNAME", "checkpoints")
            ckpt_path = os.path.join(run_dir, ckpt_dirname, "last.pt")
        if os.path.exists(ckpt_path):
            print(f"🔁 Resuming from checkpoint: {ckpt_path}")
            trainer.load_checkpoint(
                ckpt_path,
                reset_optimizer=bool(args.reset_optim),
                reset_scheduler=bool(args.reset_scheduler),
                reset_scaler=bool(args.reset_scaler),
            )
            resumed_from = ckpt_path
            print(f"   -> next_epoch={trainer.current_epoch} | best_acc={trainer.best_acc:.2f}%")
        else:
            # Fallback: weights-only last (optimizer reset)
            last_w = os.path.join(run_dir, f"{exp_name}_last.pth")
            if os.path.exists(last_w):
                print(f"⚠️ Full ckpt not found. Loading weights-only: {last_w} (optimizer/scheduler reset)")
                model.load_state_dict(torch.load(last_w, map_location=config.DEVICE))
                resumed_from = last_w
            else:
                print(f"⚠️ Resume requested but checkpoint not found: {ckpt_path}")

    run_meta.setdefault("resume", {})
    run_meta["resume"] = {
        "enabled": bool(args.resume or args.resume_path),
        "resumed_from": resumed_from,
        "reset_optim": bool(args.reset_optim),
        "reset_scheduler": bool(args.reset_scheduler),
        "reset_scaler": bool(args.reset_scaler),
    }
    dump_json(run_meta_path, run_meta)

    # 8) Save-on-signal (SIGTERM/SIGINT)
    def _handle_signal(sig, frame):
        try:
            print(f"\n⏸️ Received signal {sig}. Saving checkpoint then exit...")
            trainer.save_last_weights()
            trainer.save_checkpoint("last", extra={"signal": int(sig)})
            print("✅ Checkpoint saved.")
        except Exception as e:
            print(f"⚠️ Failed to save checkpoint on signal: {e}")
        finally:
            sys.stdout = _stdout
            sys.stderr = _stderr
            try:
                log_f.close()
            except Exception:
                pass
        raise SystemExit(0)

    try:
        import signal
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
    except Exception:
        pass

    # ----------------------------
    # Fit + collect history (best-effort)
    # ----------------------------
    try:
        trainer.fit()
    finally:
        ended_at = _now_iso()
        duration_sec = round(time.time() - t0, 2)

        # Update session info
        try:
            sess = run_meta.get("sessions", [])
            if sess:
                sess[-1]["ended_at"] = ended_at
                sess[-1]["duration_sec"] = duration_sec
        except Exception:
            pass

        run_meta["ended_at"] = ended_at
        run_meta["duration_sec"] = duration_sec

        # artifacts: weights + full checkpoints
        best_model_path = os.path.join(config.OUTPUT_DIR, f"{exp_name}_best.pth")
        last_model_path = os.path.join(config.OUTPUT_DIR, f"{exp_name}_last.pth")
        ckpt_dirname = getattr(config, "CKPT_DIRNAME", "checkpoints")
        best_full_path = os.path.join(config.OUTPUT_DIR, ckpt_dirname, "best.pt")
        last_full_path = os.path.join(config.OUTPUT_DIR, ckpt_dirname, "last.pt")

        run_meta.setdefault("artifacts", {})
        if os.path.exists(best_model_path):
            run_meta.setdefault("best", {})
            run_meta["best"]["best_ckpt"] = best_model_path
            run_meta["artifacts"]["best_weights"] = best_model_path
        if os.path.exists(last_model_path):
            run_meta["artifacts"]["last_weights"] = last_model_path
        if os.path.exists(best_full_path):
            run_meta["artifacts"]["best_full_ckpt"] = best_full_path
        if os.path.exists(last_full_path):
            run_meta["artifacts"]["last_full_ckpt"] = last_full_path

        run_meta["artifacts"]["run_meta"] = run_meta_path
        run_meta["artifacts"]["config"] = cfg_path
        run_meta["artifacts"]["console_log"] = log_path

        dump_json(run_meta_path, run_meta)

        # Summary.md
        summary_path = os.path.join(run_dir, "summary.md")
        write_summary_md(summary_path, run_meta)
        run_meta["artifacts"]["summary_md"] = summary_path
        dump_json(run_meta_path, run_meta)

        # restore stdio + close log file
        sys.stdout = _stdout
        sys.stderr = _stderr
        try:
            log_f.close()
        except Exception:
            pass

    # ----------------------------
    # Run test inference in submission mode
    # ----------------------------
    if args.submission_mode and test_loader is not None:
        print("\n" + "=" * 60)
        print("📝 GENERATING SUBMISSION FILE")
        print("=" * 60)

        best_model_path = os.path.join(config.OUTPUT_DIR, f"{exp_name}_best.pth")
        if os.path.exists(best_model_path):
            print(f"📦 Loading best checkpoint: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
        else:
            print("⚠️ No best checkpoint found, using final model weights")

        out_name = f"submission_{exp_name}_final.txt"
        trainer.predict_test(test_loader, output_filename=out_name)

        sub_path = os.path.join(config.OUTPUT_DIR, out_name)
        if os.path.exists(sub_path):
            try:
                meta = json.load(open(run_meta_path, "r", encoding="utf-8"))
            except Exception:
                meta = {}
            meta.setdefault("artifacts", {})
            meta["artifacts"]["submission_txt"] = sub_path
            dump_json(run_meta_path, meta)


if __name__ == "__main__":
    main()
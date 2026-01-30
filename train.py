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
        help="Number of training epochs (default: from config)"
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
        help="Directory to save checkpoints and submission files (default: results/)",
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
        help="Optional extra tag to append into run folder name (e.g., 'ablation1')",
    )
    parser.add_argument(
        "--backbone", 
        type=str, 
        choices=["resnet", "convnext"], 
        default=None,
        help="Backbone architecture for ResTran: 'resnet' or 'convnext'"
    )
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Initialize config with CLI overrides
    config = Config()

    # Map CLI arguments to config attributes
    arg_to_config = {
        'experiment_name': 'EXPERIMENT_NAME',
        'model': 'MODEL_TYPE',
        'epochs': 'EPOCHS',
        'batch_size': 'BATCH_SIZE',
        'learning_rate': 'LEARNING_RATE',
        'data_root': 'DATA_ROOT',
        'seed': 'SEED',
        'num_workers': 'NUM_WORKERS',
        'hidden_size': 'HIDDEN_SIZE',
        'transformer_heads': 'TRANSFORMER_HEADS',
        'transformer_layers': 'TRANSFORMER_LAYERS',
    }

    for arg_name, config_name in arg_to_config.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)

    # Special cases
    if args.aug_level is not None:
        config.AUGMENTATION_LEVEL = args.aug_level

    if args.no_stn:
        config.USE_STN = False

    if args.backbone is not None:
        config.BACKBONE_TYPE = args.backbone

    # Create per-run folder: results/<exp>_<timestamp>[_tag]/
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    exp_name = config.EXPERIMENT_NAME
    run_name = f"{exp_name}_{ts}"
    if args.run_tag:
        run_name += f"_{args.run_tag}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Output directory
    config.OUTPUT_DIR = run_dir

    # Setup console logging to file
    log_path = os.path.join(run_dir, "console.log")
    _stdout = sys.stdout
    _stderr = sys.stderr
    log_f = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(_stdout, log_f)
    sys.stderr = Tee(_stderr, log_f)

    started_at = _now_iso()
    t0 = time.time()

    seed_everything(config.SEED)

    # Run meta: initialize early (before dataset scanning)
    project_root = os.path.dirname(os.path.abspath(__file__))
    run_meta = {
        "run_id": run_name,
        "started_at": started_at,
        "ended_at": None,
        "duration_sec": None,
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "args": vars(args),
        "config": serialize_config(config),
        "system": collect_system_info(),
        "git": collect_git_info(project_root),
        "dataset": {},
        "model": {},
        "best": {},
        "artifacts": {},
    }
    run_meta_path = os.path.join(run_dir, "run_meta.json")
    cfg_path = os.path.join(run_dir, "config.json")
    dump_json(run_meta_path, run_meta)
    dump_json(cfg_path, run_meta["config"])

    print(f"🚀 Configuration:")
    print(f"   RUN_DIR : {run_dir}")
    print(f"   EXPERIMENT: {config.EXPERIMENT_NAME}")
    print(f"   MODEL: {config.MODEL_TYPE}")
    print(f"   USE_STN: {config.USE_STN}")
    print(f"   DATA_ROOT: {config.DATA_ROOT}")
    print(f"   EPOCHS: {config.EPOCHS}")
    print(f"   BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"   LEARNING_RATE: {config.LEARNING_RATE}")
    print(f"   DEVICE: {config.DEVICE}")
    print(f"   SUBMISSION_MODE: {args.submission_mode}")
    print(f"   META: {run_meta_path}")

    # Validate data path
    if not os.path.exists(config.DATA_ROOT):
        print(f"❌ ERROR: Data root not found: {config.DATA_ROOT}")
        sys.exit(1)

    # Common dataset parameters
    common_ds_params = {
        'split_ratio': config.SPLIT_RATIO,
        'img_height': config.IMG_HEIGHT,
        'img_width': config.IMG_WIDTH,
        'char2idx': config.CHAR2IDX,
        'val_split_file': config.VAL_SPLIT_FILE,
        'seed': config.SEED,
        'augmentation_level': config.AUGMENTATION_LEVEL,
    }

    # Create datasets based on mode
    if args.submission_mode:
        print("\n📌 SUBMISSION MODE ENABLED")
        print("   - Training on FULL dataset (no validation split)")
        print("   - Will generate predictions for test data after training\n")

        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='train',
            full_train=True,
            **common_ds_params
        )

        # Create test dataset if test data exists
        test_loader = None
        if os.path.exists(config.TEST_DATA_ROOT):
            test_ds = MultiFrameDataset(
                root_dir=config.TEST_DATA_ROOT,
                mode='val',
                img_height=config.IMG_HEIGHT,
                img_width=config.IMG_WIDTH,
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
                pin_memory=True
            )
        else:
            print(f"⚠️ WARNING: Test data not found at {config.TEST_DATA_ROOT}")

        val_loader = None
        val_ds = None
    else:
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='train',
            **common_ds_params
        )

        val_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='val',
            **common_ds_params
        )

        val_loader = None
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True
            )
        else:
            print("⚠️ WARNING: Validation dataset is empty.")

        test_loader = None

    if len(train_ds) == 0:
        print("❌ Training dataset is empty!")
        sys.exit(1)

    # Update meta with dataset sizes
    run_meta["dataset"] = {
        "train_size": len(train_ds),
        "val_size": (len(val_ds) if val_ds is not None else 0),
        "submission_mode": bool(args.submission_mode),
        "data_root": config.DATA_ROOT,
        "test_data_root": getattr(config, "TEST_DATA_ROOT", None),
    }
    dump_json(run_meta_path, run_meta)

    # Create training data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # Initialize model based on config
    if config.MODEL_TYPE == "restran":
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            backbone_type=config.BACKBONE_TYPE, 
        ).to(config.DEVICE)
    else:
        model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=config.USE_STN,
        ).to(config.DEVICE)

    # Print model summary
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

    # Initialize trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR
    )

    # ----------------------------
    # Fit + collect history (best-effort)
    # ----------------------------
    try:
        trainer.fit()
    finally:
        # Always try to save artifacts even if interrupted
        ended_at = _now_iso()
        duration_sec = round(time.time() - t0, 2)

        # Try to fetch history from trainer
        history = None
        if hasattr(trainer, "get_history") and callable(getattr(trainer, "get_history")):
            try:
                history = trainer.get_history()
            except Exception:
                history = None
        if history is None and hasattr(trainer, "history"):
            try:
                history = list(getattr(trainer, "history"))
            except Exception:
                history = None

        metrics_csv = os.path.join(run_dir, "metrics.csv")
        metrics_json = os.path.join(run_dir, "metrics.json")
        if history:
            dump_csv(metrics_csv, history)
            dump_json(metrics_json, {"history": history})
            run_meta["artifacts"]["metrics_csv"] = metrics_csv
            run_meta["artifacts"]["metrics_json"] = metrics_json

            # best-effort "best" extraction: pick row with max val_exact if exists, else max val_acc, else min val_loss
            best_row = None
            key_candidates = ["val_exact", "val_acc", "val_accuracy", "val_em", "val_score"]
            for k in key_candidates:
                if any((k in r) for r in history):
                    best_row = max(history, key=lambda r: (r.get(k, float("-inf"))))
                    run_meta["best"] = {"criterion": f"max({k})", **best_row}
                    break
            if best_row is None:
                if any(("val_loss" in r) for r in history):
                    best_row = min(history, key=lambda r: (r.get("val_loss", float("inf"))))
                    run_meta["best"] = {"criterion": "min(val_loss)", **best_row}
        else:
            run_meta["best"] = run_meta.get("best", {})

        # Check best checkpoint path
        best_model_path = os.path.join(config.OUTPUT_DIR, f"{exp_name}_best.pth")
        last_model_path = os.path.join(config.OUTPUT_DIR, f"{exp_name}_last.pth")
        if os.path.exists(best_model_path):
            run_meta["best"]["best_ckpt"] = best_model_path
            run_meta["artifacts"]["best_ckpt"] = best_model_path
        if os.path.exists(last_model_path):
            run_meta["artifacts"]["last_ckpt"] = last_model_path

        run_meta["ended_at"] = ended_at
        run_meta["duration_sec"] = duration_sec
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
        print("\n" + "="*60)
        print("📝 GENERATING SUBMISSION FILE")
        print("="*60)

        # Load best checkpoint if it exists
        best_model_path = os.path.join(config.OUTPUT_DIR, f"{exp_name}_best.pth")
        if os.path.exists(best_model_path):
            print(f"📦 Loading best checkpoint: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
        else:
            print("⚠️ No best checkpoint found, using final model weights")

        # Run inference on test data
        out_name = f"submission_{exp_name}_final.txt"
        trainer.predict_test(test_loader, output_filename=out_name)

        # Update meta with submission artifact
        sub_path = os.path.join(config.OUTPUT_DIR, out_name)
        if os.path.exists(sub_path):
            run_meta_path = os.path.join(config.OUTPUT_DIR, "run_meta.json")
            try:
                meta = json.load(open(run_meta_path, "r", encoding="utf-8"))
            except Exception:
                meta = {}
            meta.setdefault("artifacts", {})
            meta["artifacts"]["submission_txt"] = sub_path
            dump_json(run_meta_path, meta)


if __name__ == "__main__":
    main()

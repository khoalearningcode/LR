#!/usr/bin/env python3
"""
Infer on ICPR2026 public_test given a trained run folder or checkpoint.

Output format (one line per track):
track_id,plate_text;confidence

Examples:
  # 1) Easiest: point to run folder => auto-load config.json + pick BEST checkpoint
  python infer_public_test.py --run-dir results/restran_E6_w192_lrSim035_fd020 --data-root data/public_test

  # 2) Point to checkpoint => auto-detect run folder from checkpoint path (if possible)
  python infer_public_test.py --checkpoint results/restran_E6_w192_lrSim035_fd020/restran_best.pth --data-root data/public_test

  # 3) Manual override image size (if needed)
  python infer_public_test.py --run-dir results/restran_E6_w192_lrSim035_fd020 --img-height 32 --img-width 192
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_run_dir_from_checkpoint(ckpt: Path) -> Optional[Path]:
    ckpt = ckpt.resolve()
    for cand in [ckpt.parent, ckpt.parent.parent]:
        if (cand / "config.json").exists():
            return cand
    return None


def _pick_best_checkpoint(run_dir: Path) -> Optional[Path]:
    # Prefer "best" then fall back to "last"
    cands = [
        run_dir / "checkpoints" / "best.pt",
        run_dir / "restran_best.pth",
        run_dir / "checkpoints" / "last.pt",
        run_dir / "restran_last.pth",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def _load_state_dict(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    obj = torch.load(str(ckpt_path), map_location=device)
    if isinstance(obj, dict):
        for k in ["model_state_dict", "state_dict", "model"]:
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

    # Handle DDP "module." prefix
    if any(key.startswith("module.") for key in obj.keys()):
        obj = {k.replace("module.", "", 1): v for k, v in obj.items()}
    return obj


def build_model(config: Config) -> torch.nn.Module:
    if getattr(config, "MODEL_TYPE", "restran") == "restran":
        kwargs = dict(
            num_classes=getattr(config, "NUM_CLASSES", 37),
            transformer_heads=getattr(config, "TRANSFORMER_HEADS", 8),
            transformer_layers=getattr(config, "TRANSFORMER_LAYERS", 3),
            transformer_ff_dim=getattr(config, "TRANSFORMER_FF_DIM", 2048),
            dropout=getattr(config, "TRANSFORMER_DROPOUT", 0.1),
            use_stn=getattr(config, "USE_STN", True),
            backbone_type=getattr(config, "BACKBONE_TYPE", "convnext"),
            backbone_variant=getattr(config, "BACKBONE_VARIANT", "tiny"),
            drop_path_rate=float(getattr(config, "DROPPATH_RATE", 0.0)),
            aux_sr=False,
            frame_dropout=float(getattr(config, "FRAME_DROPOUT", 0.0)),
            backbone_pretrained=bool(getattr(config, "BACKBONE_PRETRAINED", False)),
            timm_model=str(getattr(config, "TIMM_MODEL", "") or ""),
            timm_out_index=int(getattr(config, "TIMM_OUT_INDEX", 0)),
        )
        # Filter to only args supported by your ResTranOCR signature
        import inspect
        sig = inspect.signature(ResTranOCR.__init__)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return ResTranOCR(**kwargs)
    else:
        kwargs = dict(
            num_classes=getattr(config, "NUM_CLASSES", 37),
            hidden_size=getattr(config, "HIDDEN_SIZE", 256),
            rnn_dropout=getattr(config, "RNN_DROPOUT", 0.25),
            use_stn=getattr(config, "USE_STN", True),
            frame_dropout=float(getattr(config, "FRAME_DROPOUT", 0.0)),
        )
        import inspect
        sig = inspect.signature(MultiFrameCRNN.__init__)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return MultiFrameCRNN(**kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Infer public_test from run-dir or checkpoint")
    p.add_argument("--run-dir", type=str, default=None, help="results/<run>/ folder (auto-load config + best ckpt)")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pth/.pt)")
    p.add_argument("--data-root", type=str, default="data/public_test", help="Public test root")
    p.add_argument("--out", type=str, default=None, help="Output txt path (default: <run_dir>/prediction.txt)")

    p.add_argument("--img-height", type=int, default=None)
    p.add_argument("--img-width", type=int, default=None)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    ckpt_path = Path(args.checkpoint).resolve() if args.checkpoint else None

    if run_dir is None and ckpt_path is not None:
        run_dir = _find_run_dir_from_checkpoint(ckpt_path)

    cfg_dict: Dict[str, Any] = {}
    if run_dir is not None and (run_dir / "config.json").exists():
        cfg_dict = _load_json(run_dir / "config.json")

    # Resolve checkpoint
    if ckpt_path is None:
        if run_dir is None:
            raise FileNotFoundError("Need --run-dir or --checkpoint")
        ckpt_path = _pick_best_checkpoint(run_dir)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Config
    config = Config()
    for k, v in cfg_dict.items():
        try:
            setattr(config, k, v)
        except Exception:
            pass

    # Overrides
    if args.img_height is not None:
        config.IMG_HEIGHT = int(args.img_height)
    if args.img_width is not None:
        config.IMG_WIDTH = int(args.img_width)

    # Device
    device = torch.device(args.device) if args.device else torch.device(getattr(config, "DEVICE", "cuda"))
    config.DEVICE = device

    test_root = Path(args.data_root).resolve()
    if not test_root.exists():
        raise FileNotFoundError(f"Public test root not found: {test_root}")

    out_path = Path(args.out).resolve() if args.out else ((run_dir / "prediction.txt") if run_dir else Path("prediction.txt").resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== INFER PUBLIC TEST ===")
    print(f"run_dir      : {run_dir}")
    print(f"checkpoint   : {ckpt_path}")
    print(f"public_root  : {test_root}")
    print(f"out          : {out_path}")
    print(f"model        : {getattr(config, 'MODEL_TYPE', None)}")
    print(f"backbone     : {getattr(config, 'BACKBONE_TYPE', None)}")
    print(f"img          : {getattr(config, 'IMG_HEIGHT', None)}x{getattr(config, 'IMG_WIDTH', None)}")
    print(f"device       : {config.DEVICE}")
    print("================================")

    test_ds = MultiFrameDataset(
        root_dir=str(test_root),
        mode="val",
        img_height=int(getattr(config, "IMG_HEIGHT", 32)),
        img_width=int(getattr(config, "IMG_WIDTH", 128)),
        char2idx=getattr(config, "CHAR2IDX", None),
        seed=int(getattr(config, "SEED", 42)),
        augmentation_level="light",
        input_norm=getattr(config, "INPUT_NORM", "half"),
        is_test=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(config).to(device)
    sd = _load_state_dict(ckpt_path, device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Infer"):
            images, _, _, _, _, track_ids = batch
            images = images.to(device)
            preds = model(images)
            decoded = decode_with_confidence(preds, getattr(config, "IDX2CHAR", {}))
            for i, (pred_text, conf) in enumerate(decoded):
                conf = float(max(0.0, min(1.0, conf)))
                results.append((track_ids[i], pred_text, conf))

    with open(out_path, "w", encoding="utf-8") as f:
        for track_id, pred_text, conf in results:
            f.write(f"{track_id},{pred_text};{conf:.4f}\n")

    print(f"✅ Wrote {len(results)} lines to: {out_path}")


if __name__ == "__main__":
    main()

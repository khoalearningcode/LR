#!/usr/bin/env python3
"""
Infer on ICPR2026 public_test given a trained checkpoint.

Output format (one line per track):
track_id,plate_text;confidence

Usage examples:
1) From a run folder (auto-load config + default ckpt path):
   python infer_public_test.py --run-dir results/restran_260129_231844

2) Explicit checkpoint + settings:
   python infer_public_test.py --checkpoint results/.../restran_best.pth --model restran --backbone convnext \
     --data-root data/public_test --img-height 32 --img-width 160 --out submission_public.txt
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def _maybe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def _resolve_from_run_dir(run_dir: Path) -> Tuple[Dict[str, Any], Optional[Path], Optional[Path]]:
    """
    Returns: (cfg_dict, ckpt_path_or_None, test_root_or_None)
    """
    cfg_path = run_dir / "config.json"
    meta_path = run_dir / "run_meta.json"

    cfg = _load_json(cfg_path) if cfg_path.exists() else {}
    meta = _load_json(meta_path) if meta_path.exists() else {}

    # Best ckpt path: prefer run_meta["best"]["best_ckpt"] if present
    ckpt = None
    try:
        ckpt_str = meta.get("best", {}).get("best_ckpt") or meta.get("artifacts", {}).get("best_ckpt")
        if ckpt_str:
            ckpt = Path(ckpt_str)
    except Exception:
        ckpt = None

    if ckpt is None:
        exp_name = _maybe_get(cfg, "EXPERIMENT_NAME", None)
        if exp_name:
            cand = run_dir / f"{exp_name}_best.pth"
            if cand.exists():
                ckpt = cand

    test_root = None
    # prefer config TEST_DATA_ROOT if it exists in cfg.json, else meta.dataset.test_data_root
    tr = cfg.get("TEST_DATA_ROOT") or meta.get("dataset", {}).get("test_data_root")
    if tr:
        test_root = Path(tr)

    return cfg, ckpt, test_root


def build_model(config: Config) -> torch.nn.Module:
    if config.MODEL_TYPE == "restran":
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            backbone_type=config.BACKBONE_TYPE,
            aux_sr=False,  # inference only for OCR
        )
    else:
        model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=config.USE_STN,
        )
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Infer public_test from checkpoint")
    p.add_argument("--run-dir", type=str, default=None, help="results/<run>/ folder to auto-load config + ckpt")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to *_best.pth")
    p.add_argument("--data-root", type=str, default=None, help="Public test root (default: data/public_test)")
    p.add_argument("--out", type=str, default=None, help="Output txt path (default: <run_dir>/prediction.txt or ./prediction.txt)")

    p.add_argument("--model", type=str, choices=["restran", "crnn"], default=None)
    p.add_argument("--backbone", type=str, choices=["convnext", "resnet"], default=None)
    p.add_argument("--no-stn", action="store_true")
    p.add_argument("--img-height", type=int, default=None)
    p.add_argument("--img-width", type=int, default=None)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default=None, help="cuda | cpu | cuda:0 ...")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    cfg_dict: Dict[str, Any] = {}
    auto_ckpt: Optional[Path] = None
    auto_test_root: Optional[Path] = None

    if run_dir:
        if not run_dir.exists():
            raise FileNotFoundError(f"--run-dir not found: {run_dir}")
        cfg_dict, auto_ckpt, auto_test_root = _resolve_from_run_dir(run_dir)

    # Build config object then override from cfg_dict (if any) then from CLI flags
    config = Config()

    # Override with run config.json if present
    for k, v in cfg_dict.items():
        if hasattr(config, k):
            try:
                setattr(config, k, v)
            except Exception:
                pass

    # CLI overrides
    if args.model:
        config.MODEL_TYPE = args.model
    if args.backbone:
        config.BACKBONE_TYPE = args.backbone
    if args.no_stn:
        config.USE_STN = False
    if args.img_height is not None:
        config.IMG_HEIGHT = args.img_height
    if args.img_width is not None:
        config.IMG_WIDTH = args.img_width

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = config.DEVICE
    config.DEVICE = device

    # Resolve ckpt path
    ckpt_path = Path(args.checkpoint).resolve() if args.checkpoint else auto_ckpt
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found. Provide --checkpoint or a valid --run-dir. Got: {ckpt_path}")

    # Resolve public test root
    test_root = Path(args.data_root).resolve() if args.data_root else auto_test_root
    if test_root is None:
        test_root = Path("data/public_test").resolve()
    if not test_root.exists():
        raise FileNotFoundError(f"Public test root not found: {test_root}")

    # Output path
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = (run_dir / "prediction.txt") if run_dir else Path("prediction.txt").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== INFER PUBLIC TEST ===")
    print(f"run_dir      : {run_dir}")
    print(f"checkpoint   : {ckpt_path}")
    print(f"public_root  : {test_root}")
    print(f"out          : {out_path}")
    print(f"model        : {config.MODEL_TYPE}")
    print(f"backbone     : {getattr(config, 'BACKBONE_TYPE', None)}")
    print(f"use_stn      : {config.USE_STN}")
    print(f"img          : {config.IMG_HEIGHT}x{config.IMG_WIDTH}")
    print(f"device       : {config.DEVICE}")
    print("================================")

    # Dataset + loader (test mode)
    test_ds = MultiFrameDataset(
        root_dir=str(test_root),
        mode="val",
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        seed=config.SEED,
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

    # Model
    model = build_model(config).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Infer"):
            # collate returns 6 items: images, hr_images, targets, target_lengths, labels_text, track_ids
            images, _, _, _, _, track_ids = batch
            images = images.to(device)
            preds = model(images)  # [B, T, C] log-softmax
            decoded = decode_with_confidence(preds, config.IDX2CHAR)

            for i, (pred_text, conf) in enumerate(decoded):
                # clamp conf into [0,1] just in case
                conf = float(max(0.0, min(1.0, conf)))
                results.append((track_ids[i], pred_text, conf))

    # Save
    with open(out_path, "w", encoding="utf-8") as f:
        for track_id, pred_text, conf in results:
            f.write(f"{track_id},{pred_text};{conf:.4f}\n")

    print(f"✅ Wrote {len(results)} lines to: {out_path}")


if __name__ == "__main__":
    main()
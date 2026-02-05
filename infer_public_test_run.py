#!/usr/bin/env python3
"""
Infer ICPR2026 public_test from a run directory or a checkpoint.

✅ Features
- If --run-dir is given:
  * auto-loads <run_dir>/config.json (IMG_W/H, MODEL_TYPE, BACKBONE_TYPE, USE_STN, CHAR2IDX/IDX2CHAR)
  * auto-picks BEST checkpoint in this order:
      1) run_meta.json -> best.best_ckpt / artifacts.best_ckpt (if exists)
      2) <run_dir>/<EXPERIMENT_NAME>_best.pth   (e.g., restran_best.pth)
      3) <run_dir>/checkpoints/best.pt
      4) <run_dir>/<EXPERIMENT_NAME>_last.pth
      5) <run_dir>/checkpoints/last.pt
  * default output: <run_dir>/prediction.txt

- If --checkpoint is given:
  * will try to infer run_dir from checkpoint parent (to load config.json), else falls back to defaults.

Output format (one line per track):
track_id,plate_text;confidence

Example:
python infer_public_test_run.py --run-dir results/restran_E6_w192_lrSim035_fd020 --data-root data/public_test
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


def _fix_idx2char_keys(idx2char: Any) -> Any:
    """config.json often stores IDX2CHAR keys as strings; convert to int keys when possible."""
    if not isinstance(idx2char, dict) or not idx2char:
        return idx2char
    fixed = {}
    ok = True
    for k, v in idx2char.items():
        try:
            fixed[int(k)] = v
        except Exception:
            ok = False
            break
    return fixed if ok and fixed else idx2char


def _infer_run_dir_from_ckpt(ckpt: Path) -> Optional[Path]:
    if not ckpt.exists():
        return None
    parent = ckpt.parent
    if (parent / "config.json").exists():
        return parent
    if parent.name == "checkpoints" and (parent.parent / "config.json").exists():
        return parent.parent
    return None


def _resolve_ckpt_from_run_dir(run_dir: Path, cfg: Dict[str, Any], meta: Dict[str, Any]) -> Optional[Path]:
    # 1) from meta
    try:
        ckpt_str = meta.get("best", {}).get("best_ckpt") or meta.get("artifacts", {}).get("best_ckpt")
        if ckpt_str:
            p = Path(ckpt_str)
            if p.exists():
                return p
    except Exception:
        pass

    exp = cfg.get("EXPERIMENT_NAME", "restran")
    candidates = [
        run_dir / f"{exp}_best.pth",
        run_dir / "checkpoints" / "best.pt",
        run_dir / f"{exp}_last.pth",
        run_dir / "checkpoints" / "last.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _extract_state_dict(obj: Any) -> Dict[str, Any]:
    """Support both weights-only (.pth) and full checkpoints (.pt with {'model'/'state_dict'})."""
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        # weights-only state_dict
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise ValueError("Unsupported checkpoint format (expected state_dict or dict with 'model'/'state_dict').")


def build_model(config: Config) -> torch.nn.Module:
    if config.MODEL_TYPE == "restran":
        return ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            backbone_type=config.BACKBONE_TYPE,
            aux_sr=False,
            # optional scaling knobs (if your ResTranOCR supports them)
            cnn_channels=getattr(config, "CNN_CHANNELS", 512),
            fusion_type=getattr(config, "FUSION_TYPE", "attn"),
            frame_dropout=getattr(config, "FRAME_DROPOUT", 0.0),
            drop_path_rate=getattr(config, "DROPPATH_RATE", 0.0),
            temporal_heads=getattr(config, "TEMPORAL_HEADS", 8),
            temporal_layers=getattr(config, "TEMPORAL_LAYERS", 2),
            temporal_ff_dim=getattr(config, "TEMPORAL_FF_DIM", 1024),
            temporal_dropout=getattr(config, "TEMPORAL_DROPOUT", 0.1),
        )
    return MultiFrameCRNN(
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        rnn_dropout=config.RNN_DROPOUT,
        use_stn=config.USE_STN,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Infer public_test from run-dir/checkpoint")
    p.add_argument("--run-dir", type=str, default=None, help="results/<run>/ folder (auto-load config + pick best ckpt)")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pth/.pt)")
    p.add_argument("--data-root", type=str, default=None, help="Public test root (default: data/public_test)")
    p.add_argument("--out", type=str, default=None, help="Output txt (default: <run_dir>/prediction.txt or ./prediction.txt)")

    # overrides (rarely needed if using run-dir)
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

    # If only checkpoint provided, infer run_dir to load config.json
    if run_dir is None and ckpt_path is not None:
        inferred = _infer_run_dir_from_ckpt(ckpt_path)
        if inferred is not None:
            run_dir = inferred

    cfg_dict: Dict[str, Any] = {}
    meta_dict: Dict[str, Any] = {}

    if run_dir is not None and run_dir.exists():
        cfg_path = run_dir / "config.json"
        meta_path = run_dir / "run_meta.json"
        if cfg_path.exists():
            cfg_dict = _load_json(cfg_path)
        if meta_path.exists():
            meta_dict = _load_json(meta_path)

    # Build config and override from config.json if present
    config = Config()
    for k, v in cfg_dict.items():
        if hasattr(config, k):
            try:
                setattr(config, k, v)
            except Exception:
                pass

    # Fix IDX2CHAR keys
    if hasattr(config, "IDX2CHAR"):
        config.IDX2CHAR = _fix_idx2char_keys(config.IDX2CHAR)

    # Overrides
    if args.img_height is not None:
        config.IMG_HEIGHT = args.img_height
    if args.img_width is not None:
        config.IMG_WIDTH = args.img_width

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(str(getattr(config, "DEVICE", "cuda")))
    config.DEVICE = device

    # Resolve checkpoint
    if ckpt_path is None:
        if run_dir is None:
            raise ValueError("Provide --run-dir or --checkpoint.")
        ckpt_path = _resolve_ckpt_from_run_dir(run_dir, cfg_dict, meta_dict)
        if ckpt_path is None:
            raise FileNotFoundError(f"Cannot find checkpoint in run-dir: {run_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Resolve data root
    test_root = Path(args.data_root).resolve() if args.data_root else None
    if test_root is None:
        # try from config (TEST_DATA_ROOT), else default
        tr = cfg_dict.get("TEST_DATA_ROOT")
        test_root = Path(tr).resolve() if tr else Path("data/public_test").resolve()
    if not test_root.exists():
        raise FileNotFoundError(f"Public test root not found: {test_root}")

    # Output
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = (run_dir / "prediction.txt") if run_dir is not None else Path("prediction.txt").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== INFER PUBLIC TEST ===")
    print(f"run_dir      : {run_dir}")
    print(f"checkpoint   : {ckpt_path}")
    print(f"public_root  : {test_root}")
    print(f"out          : {out_path}")
    print(f"model        : {getattr(config, 'MODEL_TYPE', None)}")
    print(f"backbone     : {getattr(config, 'BACKBONE_TYPE', None)}")
    print(f"use_stn      : {getattr(config, 'USE_STN', None)}")
    print(f"img          : {getattr(config, 'IMG_HEIGHT', None)}x{getattr(config, 'IMG_WIDTH', None)}")
    print(f"device       : {device}")
    print("================================")

    # Dataset
    test_ds = MultiFrameDataset(
        root_dir=str(test_root),
        mode="val",
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        seed=getattr(config, "SEED", args.seed),
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
    raw = torch.load(str(ckpt_path), map_location=device)
    state_dict = _extract_state_dict(raw)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Infer"):
            images, _, _, _, _, track_ids = batch
            images = images.to(device, non_blocking=True)
            preds = model(images)  # [B, T, C] log-softmax
            decoded = decode_with_confidence(preds, config.IDX2CHAR)
            for i, (pred_text, conf) in enumerate(decoded):
                conf = float(max(0.0, min(1.0, conf)))
                results.append((track_ids[i], pred_text, conf))

    with open(out_path, "w", encoding="utf-8") as f:
        for track_id, pred_text, conf in results:
            f.write(f"{track_id},{pred_text};{conf:.4f}\n")

    print(f"✅ Wrote {len(results)} lines to: {out_path}")


if __name__ == "__main__":
    main()
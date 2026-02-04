#!/usr/bin/env python3
"""
Infer on ICPR2026 public_test given a trained checkpoint OR a run directory.

✅ What this version fixes / supports:
- Chỉ cần đưa --run-dir là tự:
  + load config.json (IMG_WIDTH/IMG_HEIGHT/USE_STN/BACKBONE/…)
  + tự chọn checkpoint BEST (ưu tiên restran_best.pth, rồi checkpoints/best.pt, rồi last)
  + tự ghi output vào <run_dir>/prediction.txt

- Nếu bạn chỉ đưa --checkpoint:
  + Tự suy ra run_dir từ đường dẫn checkpoint (nếu cạnh đó có config.json)
  + Tự load config.json để không bị rơi về IMG_WIDTH=128
  + Hỗ trợ checkpoint dạng weights-only (.pth) và full checkpoint (.pt dict)

Output format (1 dòng / track):
track_id,plate_text;confidence
"""

import argparse
import json
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


# -----------------------------
# Helpers
# -----------------------------
def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_run_dir_from_checkpoint(ckpt_path: Path) -> Optional[Path]:
    """
    Infer run_dir from:
      - <run_dir>/restran_best.pth
      - <run_dir>/restran_last.pth
      - <run_dir>/checkpoints/best.pt
      - <run_dir>/checkpoints/last.pt
    """
    if not ckpt_path.exists():
        return None

    parent = ckpt_path.parent
    if (parent / "config.json").exists():
        return parent

    if parent.name == "checkpoints":
        run_dir = parent.parent
        if (run_dir / "config.json").exists():
            return run_dir

    return None


def _resolve_from_run_dir(run_dir: Path) -> Tuple[Dict[str, Any], Optional[Path], Optional[Path]]:
    """
    Returns:
      cfg_dict, ckpt_path_or_None, test_root_or_None
    """
    cfg_path = run_dir / "config.json"
    meta_path = run_dir / "run_meta.json"

    cfg = _load_json(cfg_path) if cfg_path.exists() else {}
    meta = _load_json(meta_path) if meta_path.exists() else {}

    exp_name = cfg.get("EXPERIMENT_NAME", "restran")

    # Prefer BEST then LAST
    candidates = [
        run_dir / f"{exp_name}_best.pth",
        run_dir / "checkpoints" / "best.pt",
        run_dir / f"{exp_name}_last.pth",
        run_dir / "checkpoints" / "last.pt",
    ]
    ckpt = next((p for p in candidates if p.exists()), None)

    # Prefer config TEST_DATA_ROOT if present, else meta.dataset.test_data_root
    test_root = None
    tr = cfg.get("TEST_DATA_ROOT") or meta.get("dataset", {}).get("test_data_root")
    if tr:
        test_root = Path(tr)

    return cfg, ckpt, test_root


def _fix_idx2char_keys(config: Config) -> None:
    """
    config.json thường lưu IDX2CHAR với key kiểu string ("1": "0"...).
    decode cần key int. Fix tại đây.
    """
    if not hasattr(config, "IDX2CHAR"):
        return
    m = getattr(config, "IDX2CHAR")
    if not isinstance(m, dict) or not m:
        return

    fixed: Dict[int, str] = {}
    ok = True
    for k, v in m.items():
        try:
            fixed[int(k)] = v
        except Exception:
            ok = False
            break
    if ok and fixed:
        config.IDX2CHAR = fixed


def _extract_state_dict(obj: Any) -> Dict[str, Any]:
    """
    Accept:
      - weights-only state_dict (dict[str, Tensor])
      - full checkpoint dict with "model" or "state_dict"
    """
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        # assume it's already a state_dict
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise ValueError("Unsupported checkpoint format. Expected state_dict or dict containing 'model'/'state_dict'.")


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
            aux_sr=False,  # inference only
        )
    return MultiFrameCRNN(
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        rnn_dropout=config.RNN_DROPOUT,
        use_stn=config.USE_STN,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Infer public_test from checkpoint/run-dir (fixed)")
    p.add_argument("--run-dir", type=str, default=None, help="results/<run>/ folder (auto-load config + auto-select BEST ckpt)")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pth weights or .pt full)")
    p.add_argument("--data-root", type=str, default=None, help="Public test root (default: from config TEST_DATA_ROOT or data/public_test)")
    p.add_argument("--out", type=str, default=None, help="Output txt path (default: <run_dir>/prediction.txt or ./prediction.txt)")

    p.add_argument("--model", type=str, choices=["restran", "crnn"], default=None, help="Override model type")
    p.add_argument("--backbone", type=str, choices=["convnext", "resnet"], default=None, help="Override backbone")
    p.add_argument("--no-stn", action="store_true", help="Disable STN for inference")
    p.add_argument("--img-height", type=int, default=None, help="Override IMG_HEIGHT")
    p.add_argument("--img-width", type=int, default=None, help="Override IMG_WIDTH")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default=None, help="cuda | cpu | cuda:0 ...")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    # Resolve run_dir and checkpoint
    run_dir: Optional[Path] = Path(args.run_dir).resolve() if args.run_dir else None
    ckpt_path: Optional[Path] = Path(args.checkpoint).resolve() if args.checkpoint else None

    # If only checkpoint given, infer run_dir from it (to auto-load config.json)
    if run_dir is None and ckpt_path is not None:
        inferred = _infer_run_dir_from_checkpoint(ckpt_path)
        if inferred is not None:
            run_dir = inferred

    cfg_dict: Dict[str, Any] = {}
    auto_ckpt: Optional[Path] = None
    auto_test_root: Optional[Path] = None

    if run_dir is not None:
        if not run_dir.exists():
            raise FileNotFoundError(f"--run-dir not found: {run_dir}")
        cfg_dict, auto_ckpt, auto_test_root = _resolve_from_run_dir(run_dir)

    # Need at least one of them
    if run_dir is None and ckpt_path is None:
        raise ValueError("Provide either --run-dir or --checkpoint.")

    # Build config then override from config.json
    config = Config()
    for k, v in cfg_dict.items():
        if hasattr(config, k):
            try:
                setattr(config, k, v)
            except Exception:
                pass

    # Fix IDX2CHAR keys loaded from JSON (string -> int)
    _fix_idx2char_keys(config)

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
        # config.DEVICE might be "cuda" or torch.device
        device = torch.device(str(getattr(config, "DEVICE", "cuda")))
    config.DEVICE = device

    # Pick checkpoint: explicit > auto (best)
    if ckpt_path is None:
        ckpt_path = auto_ckpt
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found. Provide --checkpoint or a valid --run-dir. Got: {ckpt_path}")

    # Resolve public test root
    if args.data_root:
        test_root = Path(args.data_root).resolve()
    elif auto_test_root is not None:
        test_root = auto_test_root.resolve()
    else:
        test_root = Path("data/public_test").resolve()

    if not test_root.exists():
        raise FileNotFoundError(f"Public test root not found: {test_root}")

    # Output path default
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = (run_dir / "prediction.txt") if run_dir is not None else Path("prediction.txt").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== INFER PUBLIC TEST (fixed) ===")
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
            # collate returns 6 items: images, hr_images, targets, target_lengths, labels_text, track_ids
            images, _, _, _, _, track_ids = batch
            images = images.to(device)
            preds = model(images)  # [B, T, C] log-softmax

            # Use fixed idx2char (int keys) to avoid empty strings
            decoded = decode_with_confidence(preds, config.IDX2CHAR)

            for i, (pred_text, conf) in enumerate(decoded):
                conf = float(max(0.0, min(1.0, conf)))
                results.append((track_ids[i], pred_text, conf))

    # Save
    with open(out_path, "w", encoding="utf-8") as f:
        for track_id, pred_text, conf in results:
            f.write(f"{track_id},{pred_text};{conf:.4f}\n")

    print(f"✅ Wrote {len(results)} lines to: {out_path}")


if __name__ == "__main__":
    main()
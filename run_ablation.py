#!/usr/bin/env python3
"""
Run backbone ablation by launching `train.py` multiple times (one run per backbone x {scratch,pretrained}).

- Chỉ ablate BACKBONE + pretrained vs scratch. Không ép img-width / lrSim / frame-dropout... (dùng đúng config.py).
- Mỗi run có RUN_DIR riêng để giữ checkpoint.
- Ghi kết quả ra 1 file duy nhất và luôn REWRITE (không bị trùng dòng) để biết đã chạy gì.
- Nếu có thư mục run_dir nhưng chưa "xong", script sẽ archive nó và chạy lại từ đầu lần sau (không resume nửa chừng).

Ví dụ:
  python run_ablation_v2.py --train-py ./train.py --output-root results/ablation_backbones

Tuỳ chọn:
  --max-runs 6           # chỉ chạy N config đầu tiên
  --only "resnet50"      # chỉ chạy backbone chứa chuỗi này
  --skip-pretrained      # chỉ scratch
  --batch-size 32        # override (0 = không override, dùng config.py)
  --save-every-steps 200 # checkpoint step-level (0 = tắt)
"""

from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ACC_PATTERNS = [
    re.compile(r"Best Val Acc:\s*([0-9]+(?:\.[0-9]+)?)\s*%?", re.IGNORECASE),
    re.compile(r"✅\s*Training complete!\s*Best Val Acc:\s*([0-9]+(?:\.[0-9]+)?)\s*%?", re.IGNORECASE),
]

DONE_MARKERS = [
    "summary.md",
    "run_meta.json",
    "restran_best.pth",
    "checkpoints/best.pt",
]

@dataclass(frozen=True)
class Exp:
    backbone: str
    pretrained: bool
    tag: str

def _now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _run(cmd: List[str], log_path: Path) -> Tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    lines: List[str] = []
    with open(log_path, "w", encoding="utf-8") as f:
        for line in proc.stdout:  # type: ignore
            lines.append(line)
            f.write(line)
    rc = proc.wait()
    return rc, "".join(lines)

def _parse_best_acc(text: str) -> Optional[float]:
    for pat in ACC_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None

def _get_help(train_py: Path) -> str:
    try:
        out = subprocess.check_output([sys.executable, str(train_py), "-h"], text=True, stderr=subprocess.STDOUT)
        return out
    except subprocess.CalledProcessError as e:
        return e.output or ""

def _has_flag(help_txt: str, flag: str) -> bool:
    return flag in help_txt


def _canonical_backbone(requested: str, backbone_choices: List[str]) -> str:
    """
    Tránh các alias mơ hồ gây crash trong code model (vd: 'resnet' -> dùng 'resnet50' nếu có).
    """
    choices = set(backbone_choices)
    if requested == "resnet" and "resnet50" in choices:
        return "resnet50"
    if requested == "convnext" and "convnext_base" in choices:
        # nếu có variant rõ ràng thì ưu tiên base
        return "convnext_base"
    return requested


def _parse_backbone_choices(help_txt: str) -> List[str]:
    m = re.search(r"--backbone\s+\{([^}]+)\}", help_txt)
    if not m:
        return []
    return [x.strip() for x in m.group(1).split(",") if x.strip()]

def _infer_run_dir(output_root: Path, tag: str) -> Path:
    # train.py: RUN_DIR = <output_root>/restran_<tag>
    return output_root / f"restran_{tag}"

def _looks_complete(run_dir: Path) -> bool:
    for rel in DONE_MARKERS:
        if (run_dir / rel).exists():
            return True
    return False

def _archive_incomplete(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    dst = run_dir.parent / f"{run_dir.name}__incomplete__{_now()}"
    shutil.move(str(run_dir), str(dst))

def _load_results(path_jsonl: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    if not path_jsonl.exists():
        return rows
    for line in path_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            tag = obj.get("tag")
            if tag:
                rows[tag] = obj
        except Exception:
            continue
    return rows

def _rewrite_results(path_jsonl: Path, rows: Dict[str, dict]) -> None:
    path_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(rows[k], ensure_ascii=False) for k in sorted(rows.keys())]
    path_jsonl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # leaderboard txt
    txt_path = path_jsonl.with_suffix(".txt")
    items = []
    for k, obj in rows.items():
        acc = obj.get("best_val_acc")
        items.append((acc if isinstance(acc, (int, float)) else -1.0, k, obj))
    items.sort(key=lambda x: x[0], reverse=True)

    out = []
    out.append("tag\tbackbone\tpretrained\tbest_val_acc\tstatus\trun_dir\tlog")
    for acc, _, obj in items:
        out.append(
            f"{obj.get('tag')}\t{obj.get('backbone')}\t{obj.get('pretrained')}\t{obj.get('best_val_acc')}\t"
            f"{obj.get('status')}\t{obj.get('run_dir')}\t{obj.get('log_path')}"
        )
    if items:
        best = items[0][2]
        out.append("")
        out.append(f"BEST\t{best.get('tag')}\tacc={best.get('best_val_acc')}")
    txt_path.write_text("\n".join(out) + "\n", encoding="utf-8")

def _build_exp_list(backbone_choices: List[str], run_pretrained: bool) -> List[Exp]:
    """
    10 backbone x 2 modes = 20 runs.
    Không dùng 'resnet' chung chung vì đã gặp lỗi Unsupported ResNet arch: resnet.
    """
    prefer = [
        # convnext family (custom code)
        "convnext_base", "convnext_small", "convnext_mid", "convnext_tiny",
        # resnet family
        "resnet101", "resnet50", "resnet34",
        "resnext101", "resnext50",
        "wide_resnet50",
    ]

    available = set(backbone_choices) if backbone_choices else set(prefer)
    chosen: List[str] = []
    for b in prefer:
        if b in available:
            chosen.append(b)

    # fallback: pick any other backbone choices (except 'resnet') to fill to 10
    for b in sorted(available):
        if b == "resnet":
            continue
        if b not in chosen:
            chosen.append(b)
        if len(chosen) >= 10:
            break
    chosen = chosen[:10]

    exps: List[Exp] = []
    for b in chosen:
        exps.append(Exp(backbone=b, pretrained=False, tag=f"ABL_{b}_scratch"))
        if run_pretrained:
            exps.append(Exp(backbone=b, pretrained=True, tag=f"ABL_{b}_pre"))
    return exps

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-py", type=str, default="train.py")
    p.add_argument("--output-root", type=str, default="results/ablation_backbones")
    p.add_argument("--results-file", type=str, default="", help="default: <output_root>/ablation_results.jsonl")
    p.add_argument("--max-runs", type=int, default=0)
    p.add_argument("--only", type=str, default="")
    p.add_argument("--skip-pretrained", action="store_true")
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-every-steps", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    train_py = Path(args.train_py).resolve()
    if not train_py.exists():
        raise FileNotFoundError(f"train.py not found: {train_py}")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    help_txt = _get_help(train_py)
    backbone_choices = _parse_backbone_choices(help_txt)

    has_pretrained_flag = _has_flag(help_txt, "--backbone-pretrained")
    run_pretrained = (has_pretrained_flag and (not args.skip_pretrained))

    exps = _build_exp_list(backbone_choices, run_pretrained=run_pretrained)
    if args.only:
        key = args.only.lower().strip()
        exps = [e for e in exps if key in e.backbone.lower()]

    if args.max_runs and args.max_runs > 0:
        exps = exps[: args.max_runs]

    results_jsonl = Path(args.results_file).resolve() if args.results_file else (output_root / "ablation_results.jsonl")
    rows = _load_results(results_jsonl)

    for exp in exps:
        run_dir = _infer_run_dir(output_root, exp.tag)
        log_path = output_root / "_ablation_logs" / f"{exp.tag}.log"

        prev = rows.get(exp.tag)
        if prev and prev.get("status") == "ok":
            print(f"[SKIP] {exp.tag} done: acc={prev.get('best_val_acc')}")
            continue

        if run_dir.exists():
            if _looks_complete(run_dir):
                # đánh dấu ok (dù best acc chưa biết)
                rows[exp.tag] = {
                    "tag": exp.tag,
                    "backbone": exp.backbone,
                    "pretrained": exp.pretrained,
                    "status": "ok",
                    "best_val_acc": prev.get("best_val_acc") if prev else None,
                    "run_dir": str(run_dir),
                    "log_path": str(log_path) if prev else None,
                    "timestamp": _now(),
                    "note": "existing markers found; skipped",
                }
                _rewrite_results(results_jsonl, rows)
                print(f"[SKIP] {exp.tag} run_dir has markers: {run_dir}")
                continue
            else:
                print(f"[ARCHIVE] incomplete -> {run_dir}")
                _archive_incomplete(run_dir)

        bb = _canonical_backbone(exp.backbone, backbone_choices)
        cmd = [sys.executable, str(train_py), "-m", "restran", "--backbone", bb, "--run-tag", exp.tag, "--output-dir", str(output_root)]
        if exp.pretrained and has_pretrained_flag:
            cmd.append("--backbone-pretrained")

        # override nhẹ (không đụng config model)
        if args.batch_size and _has_flag(help_txt, "--batch-size"):
            cmd += ["--batch-size", str(args.batch_size)]
        if args.num_workers and _has_flag(help_txt, "--num-workers"):
            cmd += ["--num-workers", str(args.num_workers)]
        if args.save_every_steps and args.save_every_steps > 0 and _has_flag(help_txt, "--save-every-steps"):
            cmd += ["--save-every-steps", str(args.save_every_steps)]

        print("\n" + "=" * 90)
        print(f"[RUN] {exp.tag} | backbone={exp.backbone} | pretrained={exp.pretrained}")
        print("cmd:", " ".join(cmd))
        print("run_dir:", run_dir)
        print("log:", log_path)
        print("=" * 90)

        if args.dry_run:
            rows[exp.tag] = {
                "tag": exp.tag,
                "backbone": exp.backbone,
                "pretrained": exp.pretrained,
                "status": "dry_run",
                "best_val_acc": None,
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "timestamp": _now(),
            }
            _rewrite_results(results_jsonl, rows)
            continue

        rc, log_text = _run(cmd, log_path)
        best_acc = _parse_best_acc(log_text)
        status = "ok" if (rc == 0 and best_acc is not None) else "fail"

        rows[exp.tag] = {
            "tag": exp.tag,
            "backbone": exp.backbone,
            "pretrained": exp.pretrained,
            "status": status,
            "best_val_acc": best_acc,
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "return_code": rc,
            "timestamp": _now(),
        }
        _rewrite_results(results_jsonl, rows)

        if status != "ok":
            print(f"[FAIL] {exp.tag} rc={rc} best_acc={best_acc} (xem log: {log_path})")

    # in best cuối cùng
    rows2 = _load_results(results_jsonl)
    best_tag, best_acc = None, -1.0
    for tag, obj in rows2.items():
        if obj.get("status") != "ok":
            continue
        acc = obj.get("best_val_acc")
        if isinstance(acc, (int, float)) and acc > best_acc:
            best_acc = acc
            best_tag = tag
    print("\n=== DONE ===")
    if best_tag:
        print(f"BEST: {best_tag} | acc={best_acc}")
    else:
        print("Chưa có run OK nào (xem _ablation_logs/).")

if __name__ == "__main__":
    main()
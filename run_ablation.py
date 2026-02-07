#!/usr/bin/env python3
"""
run_ablation.py

Chạy sweep nhiều backbone cho ResTran (MultiFrame-LPR) theo kiểu "1 backbone = 1 run_dir",
tự:
- tạo run_dir riêng
- lưu cmd + log
- nếu run bị dừng giữa chừng: lần sau sẽ tự archive run_dir đó và chạy lại từ đầu
- nếu run đã hoàn tất: skip
- sau mỗi run sẽ append kết quả Best Val Acc vào 1 file txt tổng hợp

Thiết kế để KHÔNG phụ thuộc chặt vào train.py phiên bản nào:
- Tự đọc `python train.py -h` để xem hỗ trợ flag nào (timm / backbone-variant / img-width ...)
- Chỉ add flag nếu train.py có hỗ trợ

Ví dụ:
  python run_ablation.py --preset e6 --batch-size 32 --epochs 30
  python run_ablation.py --preset bare --batch-size 64 --epochs 30
  python run_ablation.py --only convnext,resnet101 --max-runs 5
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -------------------------
# Utils: detect train.py flags
# -------------------------

def _run_train_help() -> str:
    p = subprocess.run(
        [sys.executable, "train.py", "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return p.stdout or ""


def _has_flag(help_txt: str, flag: str) -> bool:
    return flag in help_txt


def _extract_backbone_choices(help_txt: str) -> List[str]:
    # pattern like: [--backbone {a,b,c}]
    m = re.search(r"--backbone\s+\{([^}]+)\}", help_txt)
    if not m:
        return []
    raw = m.group(1)
    return [x.strip() for x in raw.split(",") if x.strip()]


# -------------------------
# Result parsing
# -------------------------

_BEST_PATTERNS = [
    re.compile(r"Best Val Acc:\s*([0-9]*\.?[0-9]+)\s*%?"),
    re.compile(r"✅\s*Training complete!\s*Best Val Acc:\s*([0-9]*\.?[0-9]+)\s*%?"),
]


def _parse_best_val_acc_from_text(text: str) -> Optional[float]:
    best = None
    for pat in _BEST_PATTERNS:
        for m in pat.finditer(text):
            try:
                v = float(m.group(1))
                best = v
            except Exception:
                continue
    return best


def _parse_best_val_acc_from_run_dir(run_dir: Path) -> Optional[float]:
    # 1) run_meta.json (nếu có)
    meta = run_dir / "run_meta.json"
    if meta.exists():
        try:
            j = json.loads(meta.read_text(encoding="utf-8"))
            # thử vài key phổ biến
            for keypath in [
                ("best", "val_acc"),
                ("best", "best_val_acc"),
                ("metrics", "best_val_acc"),
                ("metrics", "best_acc"),
            ]:
                cur = j
                ok = True
                for k in keypath:
                    if isinstance(cur, dict) and k in cur:
                        cur = cur[k]
                    else:
                        ok = False
                        break
                if ok:
                    try:
                        return float(cur) * 100.0 if (0 < float(cur) <= 1.0) else float(cur)
                    except Exception:
                        pass
        except Exception:
            pass

    # 2) summary.md (nếu có)
    summ = run_dir / "summary.md"
    if summ.exists():
        try:
            v = _parse_best_val_acc_from_text(summ.read_text(encoding="utf-8"))
            if v is not None:
                return v
        except Exception:
            pass

    # 3) train.log
    logp = run_dir / "train.log"
    if logp.exists():
        try:
            v = _parse_best_val_acc_from_text(logp.read_text(encoding="utf-8", errors="ignore"))
            if v is not None:
                return v
        except Exception:
            pass

    return None


# -------------------------
# Experiment spec
# -------------------------

class Exp:
    def __init__(
        self,
        tag: str,
        backbone: str,
        extra: List[str],
        *,
        prefer_pretrained: bool = False,
    ):
        self.tag = tag
        self.backbone = backbone
        self.extra = extra
        self.prefer_pretrained = prefer_pretrained


def _build_default_experiments(help_txt: str) -> List[Exp]:
    choices = set(_extract_backbone_choices(help_txt))

    exps: List[Exp] = []
    supports_variant = _has_flag(help_txt, "--backbone-variant")
    supports_timm = _has_flag(help_txt, "--timm-model") or ("timm" in choices)

    # ---- Built-in convnext / resnet family (non-pretrained default) ----
    if "convnext" in choices:
        if supports_variant:
            for var in ["tiny", "small", "base"]:
                exps.append(Exp(
                    tag=f"CX_{var}_scratch",
                    backbone="convnext",
                    extra=["--backbone-variant", var],
                ))
        else:
            # một số code dùng tên riêng convnext_tiny/...
            for name in ["convnext_tiny", "convnext_mid", "convnext_small", "convnext_base"]:
                if name in choices:
                    exps.append(Exp(tag=f"{name.upper()}_scratch", backbone=name, extra=[]))

    for name in ["resnet34", "resnet50", "resnet101", "resnext50", "resnext101", "wide_resnet50", "resnet"]:
        if name in choices:
            exps.append(Exp(tag=f"{name.upper()}_scratch", backbone=name, extra=[]))

    # ---- TIMM sweep (chỉ chọn conv-net / variable-size friendly) ----
    # NOTE: Những model kiểu Swin/ViT thường *bắt* input H=W=224/256 nên không hợp H=32 → skip
    # List này có thể chưa hoàn hảo; nếu model nào fail, run_ablation sẽ ghi FAIL và chạy tiếp.
    if supports_timm:
        timm_models = [
            # convnext / convnextv2
            "convnext_tiny.fb_in1k",
            "convnext_small.fb_in1k",
            "convnext_base.fb_in1k",
            "convnext_tiny.fb_in22k_ft_in1k",
            "convnext_small.fb_in22k_ft_in1k",
            "convnext_base.fb_in22k_ft_in1k",
            "convnextv2_tiny.fcmae_ft_in1k",
            "convnextv2_base.fcmae_ft_in1k",
            # regnet (thường hợp OCR khá ổn)
            "regnety_032.ra3_in1k",
            "regnety_064.ra3_in1k",
            # efficientnet v2
            "tf_efficientnetv2_s.in1k",
            "tf_efficientnetv2_m.in1k",
            "efficientnetv2_rw_s.ra2_in1k",
            "efficientnetv2_rw_m.agc_in1k",
            # repvgg / densenet
            "repvgg_b3g4",
            "densenet201.tv_in1k",
            # tresnet
            "tresnet_m.miil_in1k",
        ]

        # out-index: với conv-nets trong timm, stage 1 hoặc 2 thường ok.
        # Ta chạy 2 cấu hình cho mỗi model: out_index=1 và out_index=2 (ít nhưng đủ)
        for mname in timm_models:
            exps.append(Exp(
                tag=f"TIMM_{mname.replace('.', '_')}_o1",
                backbone="timm",
                extra=["--timm-model", mname, "--timm-out-index", "1"],
                prefer_pretrained=True,   # timm pretrained thường hữu ích; user có thể tắt bằng flag
            ))
            exps.append(Exp(
                tag=f"TIMM_{mname.replace('.', '_')}_o2",
                backbone="timm",
                extra=["--timm-model", mname, "--timm-out-index", "2"],
                prefer_pretrained=True,
            ))

    # Dedup tag
    seen = set()
    uniq = []
    for e in exps:
        if e.tag in seen:
            continue
        seen.add(e.tag)
        uniq.append(e)

    return uniq


# -------------------------
# Runner
# -------------------------

def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _archive_dir(p: Path) -> None:
    if not p.exists():
        return
    newp = p.parent / f"{p.name}__INCOMPLETE__{_timestamp()}"
    p.rename(newp)


def _stream_subprocess(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        return proc.wait()


def _maybe_add(cmd: List[str], help_txt: str, flag: str, value: Optional[str] = None, *, is_bool: bool = False) -> None:
    if not _has_flag(help_txt, flag):
        return
    if is_bool:
        cmd.append(flag)
    else:
        if value is None:
            return
        cmd.extend([flag, value])


def main() -> None:
    ap = argparse.ArgumentParser("Wide backbone ablation runner")
    ap.add_argument("--output-root", type=str, default="results/ablation", help="Thư mục gốc lưu các run")
    ap.add_argument("--preset", type=str, choices=["bare", "e6"], default="e6",
                    help="bare: chỉ backbone + restran. e6: dùng cấu hình mạnh nhất bạn đã có (w192, lrSim=0.35, fd=0.2)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="Override LR nếu muốn (nếu train.py có --lr)")
    ap.add_argument("--max-runs", type=int, default=None, help="Giới hạn số run trong 1 lần chạy")
    ap.add_argument("--only", type=str, default=None, help="Chỉ chạy những backbone/tag có chứa các token này, phân tách bằng dấu phẩy")
    ap.add_argument("--skip", type=str, default=None, help="Bỏ qua những backbone/tag có chứa các token này, phân tách bằng dấu phẩy")
    ap.add_argument("--include-pretrained", action="store_true",
                    help="Nếu bật: với exp.prefer_pretrained=True thì sẽ thêm --backbone-pretrained (nếu train.py có flag đó)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    help_txt = _run_train_help()
    if not help_txt.strip():
        print("❌ Không đọc được help từ train.py. Hãy chạy ở thư mục có train.py.", file=sys.stderr)
        sys.exit(2)

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    exps = _build_default_experiments(help_txt)

    only_tokens = [t.strip() for t in (args.only.split(",") if args.only else []) if t.strip()]
    skip_tokens = [t.strip() for t in (args.skip.split(",") if args.skip else []) if t.strip()]

    def _match_tokens(s: str, tokens: List[str]) -> bool:
        s_low = s.lower()
        return any(t.lower() in s_low for t in tokens)

    filtered: List[Exp] = []
    for e in exps:
        key = f"{e.tag}::{e.backbone}::{ ' '.join(e.extra) }"
        if only_tokens and not _match_tokens(key, only_tokens):
            continue
        if skip_tokens and _match_tokens(key, skip_tokens):
            continue
        filtered.append(e)

    if args.max_runs is not None:
        filtered = filtered[: max(0, args.max_runs)]

    results_path = out_root / "ablation_results.txt"
    done_marker = "DONE_ABLATION"

    # Load existing results (optional)
    existing: Dict[str, float] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            # format: TAG\tBEST
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    existing[parts[0].strip()] = float(parts[1].strip())
                except Exception:
                    pass

    all_rows: List[Tuple[str, str]] = []
    for tag, v in existing.items():
        all_rows.append((tag, f"{v:.4f}"))

    print("\n==============================")
    print(f"[ABLA] Found {len(filtered)} experiments to run.")
    print(f"[ABLA] Output root: {out_root}")
    print("==============================\n")

    for idx, exp in enumerate(filtered, 1):
        run_dir = out_root / exp.tag
        marker_path = run_dir / done_marker

        if marker_path.exists():
            # đã xong
            best = _parse_best_val_acc_from_run_dir(run_dir)
            if best is not None:
                print(f"[ABLA][{idx}/{len(filtered)}] SKIP (DONE) {exp.tag} | Best Val Acc: {best:.4f}")
            else:
                print(f"[ABLA][{idx}/{len(filtered)}] SKIP (DONE) {exp.tag}")
            continue

        if run_dir.exists():
            # incomplete -> archive
            print(f"[ABLA][{idx}/{len(filtered)}] Archive incomplete run: {run_dir}")
            _archive_dir(run_dir)

        run_dir.mkdir(parents=True, exist_ok=True)

        # base cmd
        cmd = [sys.executable, "train.py", "-m", "restran"]

        # backbone
        _maybe_add(cmd, help_txt, "--backbone", exp.backbone)

        # preset
        if args.preset == "e6":
            _maybe_add(cmd, help_txt, "--aug-level", "full")
            _maybe_add(cmd, help_txt, "--epochs", str(args.epochs))
            _maybe_add(cmd, help_txt, "--img-width", "192")
            _maybe_add(cmd, help_txt, "--train-lr-sim-p", "0.35")
            _maybe_add(cmd, help_txt, "--frame-dropout", "0.20")
        else:  # bare
            _maybe_add(cmd, help_txt, "--aug-level", "full")
            _maybe_add(cmd, help_txt, "--epochs", str(args.epochs))
            # không ép img-width/lrSim/fd nếu train.py có default

        # batch size
        _maybe_add(cmd, help_txt, "--batch-size", str(args.batch_size))

        # lr override
        if args.lr is not None:
            _maybe_add(cmd, help_txt, "--lr", str(args.lr))

        # always keep run-tag for readability
        _maybe_add(cmd, help_txt, "--run-tag", exp.tag)

        # force run-dir for determinism
        _maybe_add(cmd, help_txt, "--run-dir", str(run_dir))

        # saving strategy
        _maybe_add(cmd, help_txt, "--save-every-steps", "200")
        _maybe_add(cmd, help_txt, "--save-every-epochs", "1")

        # overwrite (clean start)
        _maybe_add(cmd, help_txt, "--overwrite", is_bool=True)

        # pretrained (optional)
        if args.include_pretrained and exp.prefer_pretrained:
            _maybe_add(cmd, help_txt, "--backbone-pretrained", is_bool=True)

        # extra exp-specific flags
        cmd.extend(exp.extra)

        # dump cmd
        (run_dir / "cmd.txt").write_text("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n", encoding="utf-8")

        print("\n--------------------------------")
        print(f"[ABLA][{idx}/{len(filtered)}] RUN {exp.tag}")
        print(f"[ABLA] run_dir: {run_dir}")
        print("[ABLA] cmd:")
        print("  " + " ".join(shlex.quote(x) for x in cmd))
        print("--------------------------------\n")

        if args.dry_run:
            continue

        rc = _stream_subprocess(cmd, run_dir / "train.log")

        if rc != 0:
            print(f"[ABLA] ❌ FAIL {exp.tag} (exit={rc})")
            # mark fail
            (run_dir / "FAIL").write_text(f"exit_code={rc}\n", encoding="utf-8")
            # update results file with FAIL line
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(f"{exp.tag}\tFAIL\n")
            continue

        best = _parse_best_val_acc_from_run_dir(run_dir)
        if best is None:
            # try parse from log content
            try:
                best = _parse_best_val_acc_from_text((run_dir / "train.log").read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                best = None

        if best is None:
            print(f"[ABLA] ⚠️ DONE {exp.tag} nhưng không parse được Best Val Acc")
            (run_dir / done_marker).write_text("done\n", encoding="utf-8")
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(f"{exp.tag}\tNA\n")
            continue

        print(f"[ABLA] ✅ DONE {exp.tag} | Best Val Acc: {best:.4f}")
        (run_dir / done_marker).write_text(f"best_val_acc={best}\n", encoding="utf-8")

        with open(results_path, "a", encoding="utf-8") as f:
            f.write(f"{exp.tag}\t{best:.4f}\n")

    # final summary
    rows: List[Tuple[str, float]] = []
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].strip().upper() not in {"FAIL", "NA"}:
                try:
                    rows.append((parts[0].strip(), float(parts[1].strip())))
                except Exception:
                    pass

    if rows:
        rows.sort(key=lambda x: x[1], reverse=True)
        best_tag, best_acc = rows[0]
        print("\n==============================")
        print(f"[ABLA] BEST: {best_tag} | {best_acc:.4f}")
        print("==============================\n")
    else:
        print("\n[ABLA] Chưa có kết quả hợp lệ trong ablation_results.txt\n")


if __name__ == "__main__":
    main()
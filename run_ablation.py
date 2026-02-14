#!/usr/bin/env python3
"""Run wide backbone ablation for ICPR2026 LRLPR (ResTran + STN).

Mục tiêu
- Quét ~20 backbone (ưu tiên không ConvNeXt), mỗi backbone chạy 2 chế độ:
  (1) from-scratch  (2) pretrained (nếu train.py hỗ trợ --backbone-pretrained)
- Không override img-width / lrSim / frame-dropout mặc định (lấy theo configs/config.py).
- Mỗi run tự lưu checkpoint theo cơ chế của train.py.
- Mỗi run có run-tag + log riêng.
- Luôn cập nhật 1 file kết quả (TSV) + 1 file summary (TXT) để biết đã chạy gì và điểm ra sao.
- Nếu một run bị dừng giữa chừng: lần sau script sẽ archive folder run đó và chạy lại từ đầu backbone đó.

Chạy nhanh
  python run_ablation.py --output-root results/ablation_base --epochs 30 --batch-size 32

Gợi ý quota GPU
  python run_ablation.py --max-runs 6 --output-root results/ablation_base

Filter
  python run_ablation.py --only regnet,eff --use-timm

Lưu ý
- Script sẽ tự đọc `python train.py -h` để biết train.py có hỗ trợ timm / pretrained / output-dir hay không.
- Nếu bạn bật --use-timm, script sẽ cố lấy đúng 20 model timm có pretrained (convnets, tránh ViT/Swin).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd(cmd: List[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD:\n" + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
        p.wait()
        f.write(f"\n[EXIT_CODE] {p.returncode}\n")
    return int(p.returncode)


def parse_train_help(train_py: Path) -> str:
    cmd = [sys.executable, str(train_py), "-h"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return (p.stdout or "") + "\n" + (p.stderr or "")


def parse_backbone_choices(help_text: str) -> List[str]:
    m = re.search(r"--backbone\s+\{([^}]+)\}", help_text)
    if not m:
        return []
    raw = m.group(1).strip()
    return [x.strip() for x in raw.split(",") if x.strip()]


def has_flag(help_text: str, flag: str) -> bool:
    return flag in help_text


def safe_tag(s: str) -> str:
    s = s.replace("/", "_").replace(":", "_").replace(".", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120]


def extract_best_val_acc(log_path: Path) -> Optional[float]:
    if not log_path.exists():
        return None
    txt = log_path.read_text(encoding="utf-8", errors="ignore")

    # ưu tiên "Best Val Acc" ở cuối
    pats = [
        r"Best Val Acc:\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        r"✅ Training complete! Best Val Acc:\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        r"Training complete! Best Val Acc:\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        r"Best Val Acc:\s*([0-9]+(?:\.[0-9]+)?)\b",
    ]
    for pat in pats:
        ms = re.findall(pat, txt)
        if ms:
            try:
                return float(ms[-1])
            except Exception:
                pass

    # fallback: max Val Acc
    vals = re.findall(r"Val Acc:\s*([0-9]+(?:\.[0-9]+)?)\s*%", txt)
    if vals:
        try:
            return max(float(v) for v in vals)
        except Exception:
            return None
    return None


def write_results_tsv(results_tsv: Path, row: Dict[str, str]) -> None:
    results_tsv.parent.mkdir(parents=True, exist_ok=True)
    exists = results_tsv.exists()
    with open(results_tsv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time", "run_tag", "kind", "backbone", "mode",
                "status", "best_val_acc", "run_dir", "log_path",
            ],
            delimiter="\t",
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_existing(results_tsv: Path) -> Dict[str, Dict[str, str]]:
    if not results_tsv.exists():
        return {}
    last: Dict[str, Dict[str, str]] = {}
    with open(results_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r.get("run_tag"):
                last[r["run_tag"]] = r
    return last


def rewrite_summary(results_tsv: Path, summary_txt: Path) -> None:
    if not results_tsv.exists():
        return
    rows: List[Tuple[float, Dict[str, str]]] = []
    with open(results_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r.get("status") != "DONE":
                continue
            try:
                acc = float(r.get("best_val_acc") or "nan")
            except Exception:
                continue
            if acc == acc:
                rows.append((acc, r))

    rows.sort(key=lambda x: x[0], reverse=True)
    lines = [
        f"ABRATION SUMMARY | updated: {now_stamp()}",
        f"source: {results_tsv}",
        "",
    ]
    if not rows:
        lines.append("No DONE runs yet.")
    else:
        best = rows[0][1]
        lines.append(
            f"BEST: {best['run_tag']} | {best['best_val_acc']}% | {best['kind']} | {best['backbone']} | {best['mode']}"
        )
        lines.append("")
        lines.append("TOP 20:")
        for i, (acc, r) in enumerate(rows[:20], 1):
            lines.append(
                f"{i:02d}. {r['run_tag']}\t{acc:.2f}%\t{r['kind']}\t{r['backbone']}\t{r['mode']}"
            )
    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_plan_builtin(help_text: str, include_convnext: bool) -> List[Dict[str, str]]:
    choices = parse_backbone_choices(help_text)
    if not choices:
        return []
    plan = []
    for bb in choices:
        if (not include_convnext) and ("convnext" in bb.lower()):
            continue
        plan.append({"kind": "builtin", "backbone": bb})
    return plan


def build_plan_timm_20() -> List[Dict[str, str]]:
    # 20 convnets/regnets/etc (variable input size friendly)
    base = [
        "regnety_016.ra3_in1k",
        "regnety_032.ra3_in1k",
        "regnety_064.ra3_in1k",
        "regnety_080.ra3_in1k",
        "regnetx_032.ra3_in1k",
        "regnetx_064.ra3_in1k",
        "resnet50.a1_in1k",
        "resnet101.a1h_in1k",
        "resnet152.a1_in1k",
        "resnext50_32x4d.a1h_in1k",
        "resnext101_32x8d.a1h_in1k",
        "wide_resnet50_2.racm_in1k",
        "efficientnetv2_rw_s.ra2_in1k",
        "efficientnetv2_rw_m.agc_in1k",
        "tf_efficientnet_b3.ns_jft_in1k",
        "tf_efficientnet_b4.ns_jft_in1k",
        "tf_efficientnetv2_s.in1k",
        "tf_efficientnetv2_m.in1k",
        "mobilenetv3_large_100.ra3_in1k",
        "mobilenetv3_small_100.lamb_in1k",
    ]

    # nếu có timm, lọc các model có pretrained để đỡ phí quota
    try:
        import timm  # type: ignore

        avail = set(timm.list_models(pretrained=True))
        keep = [m for m in base if m in avail]

        if len(keep) < 20:
            pool = []
            for pat in [
                "regnet*",
                "resnet*",
                "resnext*",
                "wide_resnet*",
                "*efficientnetv2*",
                "tf_efficientnet*",
                "mobilenetv3*",
            ]:
                pool.extend(timm.list_models(pat, pretrained=True))

            seen = set(keep)
            extra = []
            for m in pool:
                ml = m.lower()
                if any(k in ml for k in [
                    "vit", "swin", "deit", "beit", "cait", "pvt", "eva", "convnext", "maxvit", "coat"
                ]):
                    continue
                if m not in seen:
                    extra.append(m)
                    seen.add(m)

            keep.extend(extra[: max(0, 20 - len(keep))])
        base = keep[:20]
    except Exception:
        base = base[:20]

    return [{"kind": "timm", "timm_model": m, "timm_out_index": "3"} for m in base]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Wide backbone ablation runner")
    p.add_argument("--output-root", type=str, default="results/ablation_wide")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)

    p.add_argument("--max-runs", type=int, default=0)
    p.add_argument("--only", type=str, default=None)
    p.add_argument("--include-convnext", action="store_true")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--scratch-only", action="store_true")
    g.add_argument("--pretrained-only", action="store_true")

    p.add_argument("--use-timm", action="store_true")
    p.add_argument("--rerun", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    train_py = root / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"train.py not found next to run_ablation.py: {train_py}")

    help_text = parse_train_help(train_py)
    supports_output_dir = has_flag(help_text, "--output-dir")
    supports_overwrite = has_flag(help_text, "--overwrite")
    supports_pretrained = has_flag(help_text, "--backbone-pretrained")
    supports_timm = has_flag(help_text, "--timm-model") or has_flag(help_text, "--timm-out-index")
    supports_input_norm = has_flag(help_text, "--input-norm")

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    results_tsv = out_root / "ablation_results.tsv"
    summary_txt = out_root / "ablation_summary.txt"
    existing = load_existing(results_tsv)

    plan: List[Dict[str, str]] = []
    plan.extend(build_plan_builtin(help_text, include_convnext=args.include_convnext))

    if args.use_timm:
        if not supports_timm:
            print("[WARN] train.py không có timm args trong -h, bỏ qua timm plan.")
        else:
            plan.extend(build_plan_timm_20())

    if not plan:
        raise RuntimeError("Không tìm được backbone choices trong train.py -h và/hoặc timm plan không bật.")

    tokens = [t.strip().lower() for t in (args.only.split(",") if args.only else []) if t.strip()]
    if tokens:
        def keep(item: Dict[str, str]) -> bool:
            name = (item.get("backbone") or item.get("timm_model") or "").lower()
            return any(tok in name for tok in tokens)
        plan = [x for x in plan if keep(x)]

    if args.scratch_only:
        modes = ["scratch"]
    elif args.pretrained_only:
        modes = ["pretrained"]
    else:
        modes = ["scratch", "pretrained"]

    jobs: List[Tuple[str, Dict[str, str]]] = []
    for item in plan:
        for mode in modes:
            if mode == "pretrained" and (not supports_pretrained):
                continue
            if item["kind"] == "builtin":
                tag = safe_tag(f"ABL_{item['backbone']}__{mode}")
            else:
                tag = safe_tag(f"ABL_TIMM_{item['timm_model']}__{mode}")
            jobs.append((tag, {**item, "mode": mode}))

    if args.max_runs and args.max_runs > 0:
        jobs = jobs[: args.max_runs]

    print("=== ABLATION ===")
    print(f"output_root: {out_root}")
    print(f"jobs       : {len(jobs)}")
    print(f"supports_pretrained: {supports_pretrained}")
    print(f"supports_timm      : {supports_timm}")
    print("=============")

    for i, (tag, spec) in enumerate(jobs, 1):
        if (not args.rerun) and (tag in existing and existing[tag].get("status") == "DONE"):
            print(f"[SKIP DONE] {tag} ({existing[tag].get('best_val_acc')}%)")
            continue

        log_path = out_root / "_logs" / f"{tag}.log"

        # archive incomplete folders that contain tag (so next run starts clean)
        for p in out_root.glob(f"*{tag}*"):
            if p.is_dir() and not (p / "DONE_ABLATION").exists():
                try:
                    p.rename(p.with_name(p.name + f"__INCOMPLETE__{now_stamp()}"))
                except Exception:
                    pass

        cmd = [sys.executable, str(train_py), "-m", "restran"]

        if spec["kind"] == "builtin":
            cmd += ["--backbone", spec["backbone"]]
        else:
            cmd += [
                "--backbone", "timm",
                "--timm-model", spec["timm_model"],
                "--timm-out-index", spec.get("timm_out_index", "3"),
            ]

        if spec["mode"] == "pretrained":
            cmd += ["--backbone-pretrained"]
            if supports_input_norm:
                cmd += ["--input-norm", "imagenet"]

        # only override when provided
        if args.data_root:
            cmd += ["--data-root", args.data_root]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.lr is not None:
            cmd += ["--lr", str(args.lr)]

        if supports_output_dir:
            cmd += ["--output-dir", str(out_root)]
        cmd += ["--run-tag", tag]
        if supports_overwrite:
            cmd += ["--overwrite"]

        name = spec.get("backbone") or spec.get("timm_model") or ""
        print(f"\n[{i}/{len(jobs)}] RUN {tag} | {spec['kind']} | {name} | {spec['mode']}")
        rc = run_cmd(cmd, cwd=root, log_path=log_path)
        best = extract_best_val_acc(log_path)

        status = "DONE" if (rc == 0 and best is not None) else "FAIL"
        write_results_tsv(
            results_tsv,
            {
                "time": now_stamp(),
                "run_tag": tag,
                "kind": spec["kind"],
                "backbone": spec.get("backbone", spec.get("timm_model", "")),
                "mode": spec["mode"],
                "status": status,
                "best_val_acc": f"{best:.4f}" if best is not None else "",
                "run_dir": str(out_root),
                "log_path": str(log_path),
            },
        )
        rewrite_summary(results_tsv, summary_txt)

        # best-effort mark DONE in latest matching run folder
        if status == "DONE":
            cand = [p for p in out_root.glob(f"*{tag}*") if p.is_dir()]
            if cand:
                cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                try:
                    (cand[0] / "DONE_ABLATION").write_text("done\n", encoding="utf-8")
                except Exception:
                    pass

        print(f"[{status}] best_val_acc={best}")

    print("\n=== FINISH ===")
    print(f"results: {results_tsv}")
    print(f"summary : {summary_txt}")


if __name__ == "__main__":
    main()
